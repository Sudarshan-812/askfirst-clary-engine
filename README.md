# 🧠 Clary Diagnostics: Temporal Reasoning Engine

## Overview
This is a prototype temporal reasoning pipeline built for the Ask First engineering assignment. It analyzes longitudinal, unstructured conversational health data to identify hidden causal relationships between lifestyle triggers and physiological symptoms.

## Architecture: The Two-Stage Pipeline
Unlike standard single-pass LLM prompts, this engine utilizes a deterministic two-stage pipeline to prevent "lost-in-the-middle" hallucinations:
1. **Stage 1 (Event Extraction):** Parses raw conversations to build a strict, chronological trace of all symptoms, diet changes, and stressors, tagged by session ID.
2. **Stage 2 (Causal Analysis):** Evaluates the chronological trace against strict temporal rules (e.g., verifying time delays and checking for counter-evidence) to output a confidence-scored JSON of verified health patterns.

## Tech Stack
* **Frontend:** Streamlit
* **Reasoning Engine:** Google Gen AI SDK (`gemini-3.1-flash-lite-preview`)
* **Data:** Synthetic multi-month patient history (JSON)

## How to Run Locally
1. Clone this repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate it and install dependencies: `pip install -r requirements.txt`
4. Create a `.env` file in the root directory and add your key: `GOOGLE_API_KEY=your_key_here`
5. Run the app: `streamlit run app.py`