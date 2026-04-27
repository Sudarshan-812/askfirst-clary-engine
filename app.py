import streamlit as st
import json
import time
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables (grabs your Gemini key from .env)
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Clary Reasoning Engine", page_icon="🧠", layout="centered")

# --- Load Cleaned Data ---
@st.cache_data
def load_data():
    try:
        data_path = os.path.join(os.path.dirname(__file__), "data", "cleaned_dataset.json")
        with open(data_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure data/cleaned_dataset.json exists.")
        return {"users": []}

dataset = load_data()
users = {user["name"]: user for user in dataset["users"]}

# --- Sidebar UI ---
st.sidebar.title("🧠 Clary Diagnostics")
st.sidebar.write("Select a user to run temporal pattern analysis.")

if not users:
    st.sidebar.warning("No user data loaded.")
    st.stop()

selected_user_name = st.sidebar.radio("Patients", list(users.keys()))
selected_user_data = users[selected_user_name]

# --- Main Chat UI ---
st.title(f"Analysis for {selected_user_name}")
st.write(f"**Age:** {selected_user_data['age']} | **Occupation:** {selected_user_data['occupation']}")

# Initialize chat history uniquely for each user
if f"messages_{selected_user_name}" not in st.session_state:
    st.session_state[f"messages_{selected_user_name}"] = []

messages = st.session_state[f"messages_{selected_user_name}"]

# Display chat history
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- The AI Engine (Gemini 2.5 Flash) ---
def analyze_user(user_data):
    """
    Executes the temporal reasoning trace using Gemini 2.5 Flash and streams the JSON output.
    """
    # Safety check for the API key
    if not os.getenv("GOOGLE_API_KEY"):
        yield "Error: GOOGLE_API_KEY not found in .env file. Please check your .env setup."
        return

    # Initialize Gemini (Temperature 0 for strict analytical logic)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_output_tokens=2048
    )

    system_prompt = """You are an elite clinical reasoning AI. Your task is to analyze a user's conversational health history and identify hidden causal patterns between their lifestyle, environment, and symptoms. 

You must strictly adhere to the following operational protocol:

STEP 1: CHRONOLOGICAL MAPPING (Internal Monologue)
Before making any conclusions, extract every symptom, diet change, environment change, and stressor mentioned. Map these onto a chronological timeline using the timestamps. 

STEP 2: TEMPORAL REASONING
Analyze the timeline for causal relationships. 
- Prioritize temporal direction: A symptom occurring 48 hours after a diet change is a signal. A symptom occurring before invalidates it as the cause.
- Look for repetition: A pattern must occur multiple times, or have a medically sound delay.

STEP 3: JSON GENERATION
Output your final analysis STRICTLY as a JSON object containing an array of patterns. Do NOT wrap the output in markdown block ticks (like ```json). Just output raw JSON.

Schema:
{
  "patterns": [
    {
      "pattern_title": "Clear statement of cause and effect",
      "temporal_reasoning_trace": "Chronological explanation proving why this pattern is valid.",
      "confidence_score": "high, moderate, or low",
      "confidence_justification": "One sentence explaining why this connection is real."
    }
  ]
}"""

    # Format the user data for the prompt
    user_context = json.dumps(user_data["conversations"], indent=2)
    human_prompt = f"Analyze the following patient history and extract temporal patterns:\n\n{user_context}"

    ai_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    # Stream the response natively to the Streamlit UI
    for chunk in llm.stream(ai_messages):
        yield chunk.content

# --- Chat Input ---
if prompt := st.chat_input(f"Type 'Analyze {selected_user_name}' to begin"):
    # Add user message to chat history
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream the AI response
    with st.chat_message("assistant"):
        response = st.write_stream(analyze_user(selected_user_data))
    
    # Save AI response to history
    messages.append({"role": "assistant", "content": response})