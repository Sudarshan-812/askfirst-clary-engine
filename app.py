import streamlit as st
import json
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

st.set_page_config(page_title="Clary Reasoning Engine", page_icon="🧠", layout="centered")

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

dataset = load_data()
users = {user["name"]: user for user in dataset["users"]}

st.sidebar.title("🧠 Clary Diagnostics")
st.sidebar.write("Select a user to run multi-stage temporal pattern analysis.")

if not users:
    st.stop()

selected_user_name = st.sidebar.radio("Patients", list(users.keys()))
selected_user_data = users[selected_user_name]

st.title(f"Analysis for {selected_user_name}")
st.write(f"**Age:** {selected_user_data['age']} | **Occupation:** {selected_user_data['occupation']}")

if f"messages_{selected_user_name}" not in st.session_state:
    st.session_state[f"messages_{selected_user_name}"] = []

messages = st.session_state[f"messages_{selected_user_name}"]

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- THE NATIVE SDK PIPELINE (BULLETPROOF VERSION) ---
def run_reasoning_pipeline(user_data):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        yield "Error: GOOGLE_API_KEY not found in .env"
        return

    client = genai.Client(api_key=api_key)
    user_context = json.dumps(user_data["conversations"], indent=2)

    # --- STAGE 1: EVENT EXTRACTION ---
    stage_1_prompt = f"""You are Stage 1 of a clinical reasoning pipeline.
Your only job is to extract a chronological timeline of events from the user's history.
Extract every symptom, diet change, sleep pattern, and stressor. Tag each with its Session ID.
Do not make diagnoses. Just build the timeline.

Data:
{user_context}"""

    yield "### Stage 1: Chronological Event Extraction\n*Building intermediate temporal trace...*\n\n"

    # Non-streaming request (Highly resistant to 503 errors)
    response_1 = client.models.generate_content(
        model='gemini-3.1-flash-lite-preview',
        contents=stage_1_prompt,
        config=types.GenerateContentConfig(temperature=0)
    )

    extraction_trace = response_1.text

    # Fake the stream for the UI UX
    for word in extraction_trace.split():
        yield word + " "
        time.sleep(0.03)

    yield "\n\n---\n### Stage 2: Causal Analysis & Confidence Scoring\n*Applying temporal logic to extracted trace...*\n\n"

    time.sleep(2)  # Safety buffer

    # --- STAGE 2: PATTERN ANALYSIS ---
    stage_2_prompt = f"""You are Stage 2 of a clinical reasoning pipeline.
Analyze this chronological trace to identify causal patterns.

Strict Rules for Hidden Patterns:
- Look for delayed onset (e.g., severe calorie deficit causing hair fall 6 weeks later).
- Look for root drivers (e.g., is stress causing cramps, or is screen-time causing sleep-debt which causes cramps?).

Output your findings STRICTLY as a JSON object matching this schema. Do not include markdown ticks.
{{
  "patterns": [
    {{
      "pattern_title": "Clear cause and effect",
      "supporting_sessions": ["S01", "S04", "S07"],
      "counter_evidence_checked": "Explain if symptom appeared without trigger, or trigger without symptom.",
      "temporal_logic": "Explicitly define the time delay (e.g., 'Symptom X appeared consistently 48 hours after Trigger Y in sessions 1, 3, and 5').",
      "confidence_score": "high, moderate, or low"
    }}
  ]
}}

Chronological Trace to Analyze:
{extraction_trace}"""

    # Non-streaming request
    response_2 = client.models.generate_content(
        model='gemini-3.1-flash-lite-preview',
        contents=stage_2_prompt,
        config=types.GenerateContentConfig(temperature=0)
    )

    # Fake the stream for the UI UX
    for char in response_2.text:
        yield char
        time.sleep(0.005)

if prompt := st.chat_input(f"Type 'Analyze {selected_user_name}' to begin"):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(run_reasoning_pipeline(selected_user_data))
    
    messages.append({"role": "assistant", "content": response})