# Clary Reasoning Engine — Writeup

**Ask First AI Intern Assignment** · Sudarshan Kulkarni · April 2026

---

## The problem

A single LLM call asked to read three months of messy conversational health data fragments context. It anchors on recent loud signals like work stress and loses subtle long-arc patterns like nutritional deficiency causing hair fall six weeks later. My approach is a deterministic two-stage pipeline that separates extraction from inference, with strict input validation at the boundary so the system fails loudly on malformed data instead of silently degrading.

## Architecture

**Input validation.** Incoming dataset JSON is validated against a Pydantic `DatasetSchema` before any model is called. Wrong shape, missing fields, or type mismatches surface as explicit UI errors. The system constrains itself to Ask First-formatted data and tells users clearly when input doesn't conform.

**Stage 1 — Event extraction** (`gemini-3.1-flash-lite-preview`). Raw conversations become a structured timeline of events: session_id, timestamp, week_number, symptoms, lifestyle factors, contexts, severity. Flash-Lite is right here because extraction is high-volume structured work, not deep reasoning. The output is validated against `PatientTimeline` before Stage 2 sees it — schema drift fails loudly.

**Stage 2 — Causal reasoning** (`gemini-2.5-flash`). The validated timeline plus the original conversations are passed to a stronger reasoning model. Both inputs are deliberate: the structured timeline drives systematic reasoning, the raw conversations preserve nuance like "low level but constant" or "a little dizzy but I think that's normal" that structured extraction flattens.

**Stage 3 — Conversational follow-up.** Users ask questions through a chat interface. Each turn receives the detected patterns and original conversations as context, so answers cite specific session IDs and stay grounded.

## Reasoning principles enforced in Stage 2

- **Counter-evidence verification.** For every proposed pattern, the model identifies sessions where the trigger appeared but the symptom didn't. Empty counter-evidence strengthens confidence; significant counter-evidence lowers it.
- **Temporal directionality.** Causes must precede effects. A symptom appearing before a lifestyle change cannot be caused by it. This rules out correlating synchronous events.
- **Biological plausibility.** Acute triggers (late dinner causing stomach pain within hours) are distinguished from chronic accumulation (seven weeks of sleep debt producing diffuse anxiety). Lag times must match plausible mechanisms.

## Chunking and context strategy

I considered three options. Per-conversation embedding with retrieval was overkill for nine conversations per user. A sliding window over conversations is exactly what causes the context-fragmentation problem. I chose the third: full timeline plus full conversations per user, chunked at the user level. At ~3,500 tokens of history, the full picture fits comfortably and Stage 2 reasons holistically. Cross-user reasoning isn't relevant since planted patterns are intra-user. Right call at this scale; would not scale to 1000 conversations per user (see below).

## Where the system fails (the honest part)

**1. Confident hallucination of plausible mechanisms.** The system always produces a `causal_mechanism` field, even for patterns supported by only two sessions. The biological explanation reads identically whether evidence is overwhelming or thin. Fix: gate mechanism generation on minimum evidence count.

**2. Promoting one-off observations to patterns.** In USR001, a single back-pain mention became "Sedentary-Induced Musculoskeletal Strain" with a full mechanism — but it appeared in one of nine sessions. Needs an explicit floor for what counts as a pattern versus an isolated observation.

**3. Loss of qualitative nuance in Stage 1.** When Meera says she's "just a little dizzy in the mornings but I think that's normal," Stage 1 captures "dizziness" but loses the self-dismissal — which is itself diagnostic. A future Stage 1 should extract qualitative markers (denial, hedging, uncertainty) as a separate field.

**4. Free-tier API reliability.** Gemini 3 Flash returned 503 errors during peak hours. I manually fell back to Gemini 2.5 Flash to complete Meera's analysis. Production fix: automatic fallback chain wrapping streaming calls that catches 503/429/quota errors and retries with progressively more reliable models. Designed but not implemented in tonight's window.

## What I'd build differently with more time

**Split hypothesis from scoring.** Stage 2 currently does both, which pushes the model to converge on 2–3 confident patterns and stop searching. Splitting into broad hypothesis generation followed by per-pattern evidence scoring would lift recall.

**Eval harness with ground truth.** Built this without seeing planted patterns; with labels I'd measure precision/recall per architectural variant.

**Knowledge graph for scale.** At 100+ conversations per user across years, full-context breaks. Right architecture: events as nodes in a property graph (Neo4j) with temporal edges encoding lag times; LLM becomes a synthesis layer over deterministic graph queries.

---

**Repo:** [github.com/Sudarshan-812/askfirst-clary-engine](https://github.com/Sudarshan-812/askfirst-clary-engine) · **Live demo:** [sudarshan-812-askfirst-clary-engine-app-1kqvvi.streamlit.app](https://sudarshan-812-askfirst-clary-engine-app-1kqvvi.streamlit.app/)

Built without LangChain. Direct Gemini API + Pydantic schema validation.