import streamlit as st
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError, Field
from typing import List

load_dotenv()

st.set_page_config(
    page_title="Clary Reasoning Engine",
    page_icon="🧠",
    layout="centered"
)

EXTRACTION_MODEL_CHAIN = [
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

REASONING_MODEL_CHAIN = [
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
]


# ===========================================
# PYDANTIC SCHEMAS
# ===========================================

class TimelineEvent(BaseModel):
    session_id: str
    timestamp: str
    week_number: int
    symptoms_reported: List[str]
    lifestyle_factors: List[str]
    contexts: List[str]
    severity: str


class PatientTimeline(BaseModel):
    events: List[TimelineEvent]


class EvidenceItem(BaseModel):
    session_id: str
    timestamp: str
    note: str = Field(description="Brief explanation of why this session supports or counters the pattern")


class CausalPattern(BaseModel):
    pattern_title: str
    supporting_evidence: List[EvidenceItem]
    counter_evidence: List[EvidenceItem] = Field(
        description="Sessions where the trigger appeared but the symptom did NOT, OR sessions where the symptom appeared without the trigger. Empty list is valid only if rigorous check found no counter-evidence."
    )
    temporal_logic: str = Field(
        description="Specific time-based reasoning with session numbers and lag times"
    )
    causal_mechanism: str = Field(
        description="The biological or behavioral mechanism explaining why this connection is plausible"
    )
    confidence_score: str = Field(
        description="One of: very_high, high, moderate, low"
    )
    confidence_reasoning: str = Field(
        description="One sentence explaining why this confidence level was chosen"
    )


class AnalysisResult(BaseModel):
    reasoning_trace: List[str] = Field(
        description="Step-by-step thinking the system performed before reaching final patterns"
    )
    patterns: List[CausalPattern]


# ===========================================
# DATA LOADING & UTILITIES
# ===========================================

@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "cleaned_dataset.json")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Dataset not found at: {file_path}")
        return {"users": []}


def save_output(user_id: str, analysis: AnalysisResult):
    """Persist analysis output to outputs/ directory for traceability."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{user_id}_patterns.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(analysis.model_dump_json(indent=2))


def extract_json_from_text(text: str) -> str:
    """Strip markdown fences from LLM output before JSON parsing."""
    clean = text.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    return clean.strip()


def get_genai_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found in .env file")
        st.stop()
    return genai.Client(api_key=api_key)


def stream_with_fallback(client, prompt, config, model_chain, ui_placeholder=None):
    """
    Try each model in model_chain in order. On 503/429/quota errors,
    fall through to the next model. Stream tokens to ui_placeholder
    (a Streamlit st.empty() container) if provided.

    Returns: (full_response_text, model_used)
    Raises: RuntimeError if all models in chain fail.
    """
    _RETRYABLE = ["503", "429", "unavailable", "quota", "rate limit", "exhausted"]
    last_error = None

    for model_name in model_chain:
        full_text = ""
        try:
            response = client.models.generate_content_stream(
                model=model_name,
                contents=prompt,
                config=config,
            )
            for chunk in response:
                if chunk.text:
                    full_text += chunk.text
                    if ui_placeholder is not None:
                        ui_placeholder.code(full_text, language="json")
            return full_text, model_name
        except Exception as e:
            error_str = str(e).lower()
            if any(marker in error_str for marker in _RETRYABLE):
                last_error = e
                if ui_placeholder is not None:
                    ui_placeholder.warning(
                        f"`{model_name}` unavailable ({type(e).__name__}), "
                        f"trying next model..."
                    )
                time.sleep(2)
                continue
            raise

    raise RuntimeError(
        f"All models in chain failed. Last error: {last_error}"
    )


# ===========================================
# UI SETUP
# ===========================================

dataset = load_data()
users = {user["name"]: user for user in dataset.get("users", [])}

st.sidebar.title("Clary Reasoning Engine")
st.sidebar.caption("Multi-stage temporal pattern analysis")

if not users:
    st.error("No users loaded. Check data/cleaned_dataset.json")
    st.stop()

selected_user_name = st.sidebar.radio("Select Patient", list(users.keys()))
selected_user_data = users[selected_user_name]

st.sidebar.markdown("---")
st.sidebar.markdown("**Pipeline:**")
st.sidebar.markdown("**Stage 1 — Extraction**")
for m in EXTRACTION_MODEL_CHAIN:
    st.sidebar.markdown(f"- `{m}`")
st.sidebar.markdown("**Stage 2 — Reasoning**")
for m in REASONING_MODEL_CHAIN:
    st.sidebar.markdown(f"- `{m}`")
st.sidebar.caption("Auto-fallback on 503/429 errors")

st.title(f"Analysis: {selected_user_name}")
st.caption(
    f"Age {selected_user_data['age']} · "
    f"{selected_user_data['occupation']} · "
    f"{selected_user_data['location']}"
)

with st.expander("View raw conversation history"):
    st.json(selected_user_data["conversations"])


# ===========================================
# SESSION STATE — per-user analysis cache
# ===========================================

analysis_key = f"analysis_{selected_user_data['user_id']}"
chat_key = f"chat_{selected_user_data['user_id']}"

if analysis_key not in st.session_state:
    st.session_state[analysis_key] = None
if chat_key not in st.session_state:
    st.session_state[chat_key] = []


# ===========================================
# PIPELINE EXECUTION
# ===========================================

if st.button(f"Run Analysis on {selected_user_name}", type="primary"):
    client = get_genai_client()
    user_context = json.dumps(selected_user_data["conversations"], indent=2)

    # Reset chat for this user
    st.session_state[chat_key] = []

    st.markdown("---")

    # ============================================
    # STAGE 1: STRUCTURED EVENT EXTRACTION
    # ============================================
    st.markdown("### Stage 1 — Event Extraction")
    st.caption("Extracting structured timeline from raw conversations")

    stage_1_prompt = f"""You are Stage 1 of a clinical reasoning pipeline.
Your task: extract a structured timeline of events from the user's conversation history.

For each conversation, identify:
- session_id (e.g., USR001_S01)
- timestamp (ISO format from the data)
- week_number (calculate from January 1, 2026 as week 1)
- symptoms_reported (specific symptoms mentioned, e.g., "stomach pain", "afternoon headaches")
- lifestyle_factors (specific behaviors mentioned, e.g., "dinner at 11:30pm", "intermittent fasting 700 calories")
- contexts (situational factors, e.g., "work deadline", "product launch stress", "sprint review")
- severity (use the severity field from the data, or infer if missing)

Be specific. Don't summarize as "ate late" — write "dinner at 11:30pm".
Don't summarize as "stressed" — write "deadline week" or "sprint review".
Capture every distinct symptom, lifestyle factor, and context — even ones mentioned briefly.

Output strictly valid JSON matching this exact schema. Do not wrap in markdown:
{{
  "events": [
    {{
      "session_id": "USR001_S01",
      "timestamp": "2026-01-05T23:14:00",
      "week_number": 1,
      "symptoms_reported": ["stomach pain", "burning sensation"],
      "lifestyle_factors": ["dinner at 11:30pm", "long work session"],
      "contexts": ["work deadline implied"],
      "severity": "mild"
    }}
  ]
}}

Conversation data:
{user_context}"""

    stage_1_placeholder = st.empty()

    try:
        raw_stage_1, model_used_1 = stream_with_fallback(
            client=client,
            prompt=stage_1_prompt,
            config=types.GenerateContentConfig(temperature=0),
            model_chain=EXTRACTION_MODEL_CHAIN,
            ui_placeholder=stage_1_placeholder,
        )
        if model_used_1 != EXTRACTION_MODEL_CHAIN[0]:
            st.info(f"Stage 1 used fallback model: `{model_used_1}`")
    except Exception as e:
        st.error(f"All extraction models failed: {e}")
        st.stop()

    try:
        clean_json_1 = extract_json_from_text(raw_stage_1)
        timeline_data = PatientTimeline.model_validate_json(clean_json_1)
        stage_1_placeholder.empty()
    except (ValidationError, json.JSONDecodeError) as e:
        st.error(f"Stage 1 validation failed: {e}")
        st.code(raw_stage_1, language="json")
        st.stop()

    with st.expander("Stage 1 Output: Structured Timeline", expanded=False):
        for event in timeline_data.events:
            st.markdown(
                f"**Week {event.week_number}** · "
                f"`{event.session_id}` · "
                f"{event.timestamp[:16]}"
            )
            if event.symptoms_reported:
                st.markdown(f"- Symptoms: {', '.join(event.symptoms_reported)}")
            if event.lifestyle_factors:
                st.markdown(f"- Lifestyle: {', '.join(event.lifestyle_factors)}")
            if event.contexts:
                st.markdown(f"- Context: {', '.join(event.contexts)}")
            st.markdown(f"- Severity: {event.severity}")
            st.markdown("")

    st.success(f"Stage 1 complete — {len(timeline_data.events)} events extracted")

    # Brief pause to respect free-tier rate limits
    time.sleep(2)

    # ============================================
    # STAGE 2: CAUSAL REASONING
    # ============================================
    st.markdown("### Stage 2 — Causal Reasoning")
    st.caption("Identifying patterns through temporal and causal analysis")

    stage_2_prompt = f"""You are Stage 2 of a clinical reasoning pipeline.
Your task: identify causal patterns from a structured timeline of user events.

CRITICAL: Examine the timeline systematically. Most users have 3-5 distinct patterns, not just 1-2.
Look for ALL of the following pattern types:
- Recurring symptom-trigger pairs (same symptom appearing multiple times after the same lifestyle factor)
- Hierarchical patterns (where one root cause drives multiple downstream symptoms)
- Delayed-onset patterns (where the trigger and symptom are weeks apart)
- Root-cause vs proximate-cause distinctions (when multiple factors correlate, which is the consistent driver?)

Apply these reasoning principles rigorously:

1. TEMPORAL DIRECTIONALITY
Causes must precede effects. A symptom appearing BEFORE a lifestyle change cannot be caused by it.
Always verify which event came first.

2. PROXIMATE VS ROOT CAUSE
When multiple factors correlate with a symptom, identify which factor is consistently 
present across ALL instances. The factor that's always there is more likely the root driver 
than one that's only sometimes present.

3. DELAYED ONSET
Some causes have biological lag times. Nutritional deficiencies affect hair growth weeks later.
Sleep debt accumulates and produces symptoms over weeks. Don't dismiss a pattern just because 
trigger and symptom are weeks apart — check if the lag is biologically plausible for the 
proposed mechanism.

4. COUNTER-EVIDENCE CHECK
For every pattern you propose, identify sessions where the trigger appeared but the symptom 
did NOT, OR sessions where the symptom appeared without the trigger. If no counter-evidence 
exists, the pattern is stronger. If counter-evidence is significant, lower your confidence.

5. FREQUENCY AND CONFIDENCE
A pattern based on 3-4+ instances is stronger than one based on 2.
Be conservative with confidence scores. Reserve "very_high" for patterns with no counter-evidence
and a clear biological mechanism.

DO NOT STOP at 1-2 patterns. Examine every recurring symptom and every recurring lifestyle 
factor in the timeline. If a user has 4 patterns, output all 4.

Output strictly valid JSON matching this exact schema. Do not wrap in markdown:
{{
  "reasoning_trace": [
    "Step-by-step thoughts the system considered, e.g., 'Noticed stomach pain in S01, S04, S07, S09 — checked timing of dinner in those sessions'",
    "Each step is one bullet of your thinking",
    "Include observations about patterns you considered but rejected, and why"
  ],
  "patterns": [
    {{
      "pattern_title": "Clear cause-and-effect statement",
      "supporting_evidence": [
        {{
          "session_id": "USR001_S01",
          "timestamp": "2026-01-05T23:14:00",
          "note": "Specific evidence from this session"
        }}
      ],
      "counter_evidence": [
        {{
          "session_id": "USR001_S03",
          "timestamp": "2026-01-19T09:15:00",
          "note": "Trigger absent here AND symptom absent — confirms link"
        }}
      ],
      "temporal_logic": "Specific time-based reasoning with session numbers and lag times",
      "causal_mechanism": "Biological or behavioral mechanism explaining the connection",
      "confidence_score": "very_high",
      "confidence_reasoning": "One sentence justifying this confidence level"
    }}
  ]
}}

Validated structured timeline:
{timeline_data.model_dump_json(indent=2)}

Original conversations (for nuance and context the structured timeline may have lost):
{user_context}"""

    stage_2_placeholder = st.empty()

    try:
        raw_stage_2, model_used_2 = stream_with_fallback(
            client=client,
            prompt=stage_2_prompt,
            config=types.GenerateContentConfig(temperature=0),
            model_chain=REASONING_MODEL_CHAIN,
            ui_placeholder=stage_2_placeholder,
        )
        if model_used_2 != REASONING_MODEL_CHAIN[0]:
            st.info(f"Stage 2 used fallback model: `{model_used_2}`")
    except Exception as e:
        st.error(f"All reasoning models failed: {e}")
        st.stop()

    try:
        clean_json_2 = extract_json_from_text(raw_stage_2)
        analysis_data = AnalysisResult.model_validate_json(clean_json_2)
        stage_2_placeholder.empty()
    except (ValidationError, json.JSONDecodeError) as e:
        st.error(f"Stage 2 validation failed: {e}")
        st.code(raw_stage_2, language="json")
        st.stop()

    # Persist output and cache in session state
    save_output(selected_user_data["user_id"], analysis_data)
    st.session_state[analysis_key] = analysis_data


# ===========================================
# RENDER STORED ANALYSIS (persists across chat turns)
# ===========================================

if st.session_state[analysis_key] is not None:
    analysis_data = st.session_state[analysis_key]

    st.markdown("---")

    # Reasoning trace
    with st.expander("Reasoning Trace (system's thought process)", expanded=False):
        for i, step in enumerate(analysis_data.reasoning_trace, start=1):
            st.markdown(f"{i}. {step}")

    # Patterns
    st.markdown(f"### Detected Patterns ({len(analysis_data.patterns)})")

    confidence_order = {"very_high": 0, "high": 1, "moderate": 2, "low": 3}
    sorted_patterns = sorted(
        analysis_data.patterns,
        key=lambda p: confidence_order.get(p.confidence_score.lower(), 99)
    )

    for i, pattern in enumerate(sorted_patterns, start=1):
        conf_label = f"[{pattern.confidence_score.upper().replace('_', ' ')}]"
        with st.expander(
            f"{conf_label}  Pattern {i}: {pattern.pattern_title}",
            expanded=True
        ):
            st.markdown(f"**Confidence reasoning:** {pattern.confidence_reasoning}")

            st.markdown(f"**Temporal logic:**")
            st.markdown(pattern.temporal_logic)

            st.markdown(f"**Causal mechanism:**")
            st.markdown(pattern.causal_mechanism)

            st.markdown(
                f"**Supporting evidence ({len(pattern.supporting_evidence)} sessions):**"
            )
            for ev in pattern.supporting_evidence:
                st.markdown(f"- `{ev.session_id}` · {ev.timestamp[:16]} · {ev.note}")

            if pattern.counter_evidence:
                st.markdown(
                    f"**Counter-evidence checked ({len(pattern.counter_evidence)} sessions):**"
                )
                for ev in pattern.counter_evidence:
                    st.markdown(f"- `{ev.session_id}` · {ev.timestamp[:16]} · {ev.note}")
            else:
                st.markdown(
                    f"**Counter-evidence:** None found — pattern holds across all relevant sessions."
                )

    # Download button
    st.download_button(
        label=f"Download {selected_user_data['user_id']}_patterns.json",
        data=analysis_data.model_dump_json(indent=2),
        file_name=f"{selected_user_data['user_id']}_patterns.json",
        mime="application/json"
    )

    # ===========================================
    # CONVERSATIONAL INTERFACE
    # ===========================================

    st.markdown("---")
    st.markdown("### Ask Clary about these patterns")
    st.caption(
        "Ask follow-up questions about specific patterns, sessions, "
        "or how the system reached its conclusions."
    )

    # Render existing chat history
    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_question = st.chat_input(
        f"Ask about {selected_user_name}'s patterns..."
    )

    if user_question:
        st.session_state[chat_key].append({
            "role": "user",
            "content": user_question
        })
        with st.chat_message("user"):
            st.markdown(user_question)

        # Build context for follow-up
        user_context = json.dumps(selected_user_data["conversations"], indent=2)

        chat_prompt = f"""You are Clary, a thoughtful health reasoning assistant.

The user has run a pattern analysis on their health data and is now asking 
a follow-up question. Answer naturally and conversationally — do NOT output JSON.

Cite specific session IDs and timestamps when relevant. If the user asks about 
something not in the data or analysis, say so honestly.

Detected patterns from the analysis:
{analysis_data.model_dump_json(indent=2)}

Original conversation history:
{user_context}

User's question: {user_question}

Answer in 2-4 short paragraphs. Be specific and grounded in the data."""

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response_text = ""
            client = get_genai_client()
            last_error = None
            for model_name in REASONING_MODEL_CHAIN:
                try:
                    response_text = ""
                    stream = client.models.generate_content_stream(
                        model=model_name,
                        contents=chat_prompt,
                        config=types.GenerateContentConfig(temperature=0.3)
                    )
                    for chunk in stream:
                        if chunk.text:
                            response_text += chunk.text
                            placeholder.markdown(response_text)
                    if model_name != REASONING_MODEL_CHAIN[0]:
                        response_text += f"\n\n*(answered using {model_name} fallback)*"
                        placeholder.markdown(response_text)
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    if any(c in error_str for c in ["503", "429", "unavailable", "quota", "rate limit", "exhausted"]):
                        last_error = e
                        continue
                    else:
                        raise
            else:
                response_text = f"All chat models failed. Last error: {last_error}"
                placeholder.error(response_text)

        st.session_state[chat_key].append({
            "role": "assistant",
            "content": response_text
        })

else:
    st.info(
        f"Click 'Run Analysis on {selected_user_name}' above to start the pipeline."
    )