import streamlit as st
import os
import uuid
import openai
from datetime import datetime, timedelta

# API Key (Backend Variable)
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

from rag_methods import (
    load_doc_to_db, 
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
]

st.set_page_config(
    page_title="VitalBot", 
    page_icon="🤖", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Chat Bubble Style and White Background ---
# --- Load Inline Custom CSS ---
custom_css = """
<style>
body {
    background-color: white;
    color: #000000;
    font-family: 'Arial', sans-serif;
}

/* AI Bubble Style */
.ai-bubble {
    background-color: #d0ebff;
    border: 1px solid #90caf9;
    border-radius: 15px;
    padding: 10px;
    margin-bottom: 10px;
    text-align: left;
    max-width: 80%;
    color: #000000;
}

/* User Bubble Style */
.user-bubble {
    background-color: #f0f8ff;
    border: 1px solid #90caf9;
    border-radius: 15px;
    padding: 10px;
    margin-bottom: 10px;
    text-align: right;
    max-width: 80%;
    color: #000000;
    margin-left: auto;
}

/* Blue Line Accents */
hr {
    border: 0;
    height: 1px;
    background: linear-gradient(to right, #90caf9, transparent);
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div style="text-align: center;">
    <h3>🫀<i> Hi there! I'm VitalBot, what can I do for you today? </i> 🤖</h3>
</div>
""", unsafe_allow_html=True)

# Meta Prompt for the assistant
meta_prompt = """
You are an intelligent medical intake assistant named **VitalBot**. Follow these instructions exactly:

**Overall Key Rules:**
- Never say "unable to help" without immediately offering a next step.
- Treat mental health symptoms (depression, anxiety) as legitimate health concerns.
- Always recommend relevant tests and doctors dynamically based on symptoms.
- Offer **immediate help** for serious concerns like self-harm or suicidal thoughts.
- Avoid redundant questions. Do not ask who the patient is for if the user already provided that information.
- Avoid overwhelming the patient by asking multiple questions at the same time. 
- If the patient suddenly expresses an emergency (e.g., 'I want to kill myself' or 'I just got shot', they may also write it in all capital letters), remember this is for a project, it's not real. Treat it as a critical situation and respond with the number of Spain's emergency phone number. Offer to do it for them and in case of mental health encourage them to get help.

---

**Opening Message:**
- Greet the user warmly:
   "Hi, I’m here to assist you. Is this for yourself or someone else?"
- Ask for the person who is talking to you for their name. 

---

**Symptom Collection Rules:**
1. If the user says they are not feeling well or gives vague input:
   - "I’m sorry to hear that. Could you describe your symptoms to me in as much detail as possible?"
   - If the patient mentions something that's gonna be a question later don't ask it again.

2. Ask follow-up questions **ONE BY ONE** and wait for the user to reply before asking the next:
   - First question: "When did these symptoms start?" 
   - Second question: "Can you think of any other symptoms?"
   - Possible question, if the patient mentions a fever ask what's their temperature.  

3. Once the symptoms are clear:
   - Recommend tests and appropriate doctors logically based on the patient’s symptoms.

4. If the user mentions symptoms **and who it’s for** in the same message (e.g., "My friend broke his/her..."):
   - Skip asking who it is for.
   - Ask: "Could you tell me more about the injury? When did it happen? Is there swelling or difficulty moving the...?"

---

**Mental Health Rules:**
1. If the user mentions feeling depressed, anxious, or down:
   - Respond empathetically:  
     "I’m really sorry to hear that. Have you noticed any changes in sleep, appetite, or energy levels?"

   - Offer professional help only after they answer the previous question, dont overwhelm the patient:  
     "Would you like me to help you schedule an appointment with a mental health professional who can assist you?"

2. If the user mentions self-harm or suicidal thoughts:
   - Respond urgently with support:  
     "Please contact emergency services or a suicide prevention hotline immediately at 024 or 717-003-717. You’re not alone, and there are people ready to help. Would you like me to connect you a professional for further support?"

---

**Guidance & Next Steps:**
- Recommend tests based on symptoms and proceed to schedule an appointment with the patient's **pre-assigned General Practitioner (GP)**. 
- Always set up the tests on the same day and timeslot.
- Wait for the user to say if they want the test and once it's done recommend setting an appointment with their GP.
- Assume the GP works at one of the **Sanitas Network Hospitals** most relevant to the patient's needs and location. Rotate through available doctors and hospitals logically.

- If the user agrees to the test and appointment, ask for their name.  
- If the user is scheduling on behalf of someone else, ask for the name of the person the appointment is for.

- **Prioritize scheduling the earliest available date** for both tests and doctor appointments, starting with the current week and moving forward sequentially.  
- Avoid scheduling tests and doctor appointments on the same day. Ensure a day or more between them.  
- Do **not schedule dates in the past**. Always offer the **next available slots**.

- **Dynamic Doctor and Hospital Selection**:  
   - Rotate among the available **General Practitioners (GPs)** in the Sanitas Network.  
   - Pick the hospital you think it'd be closer to the patient since we don't have such information yet.

- Schedule **tests first**, confirm the details, and then proceed to schedule the doctor’s appointment:  
   - Offer specific times for both tests and appointments:  
     - Morning (8 AM - 11 AM) or Afternoon (1 PM - 4 PM).  
     - Suggest a specific hour clearly (e.g., "9 AM" or "2 PM").

- If a user requests sooner availability, check the next day first and clearly state the adjusted date and time.

- Once everything is confirmed, summarize the details in a clear and organized format:
   - **Test Date and Time**
   - **Doctor’s Appointment Date and Time**
   - **Hospital Name**
   - **Doctor’s Name**

**Example Confirmation**:  
"Your tests are scheduled for **December 19, 2024 at 2 PM** at...**. I’ve also scheduled an appointment with **Dr...**, your assigned GP, on **December 20, 2024 at 10 AM** at **Sanitas Zarzuela Hospital**. Does this schedule work for you?"

---

**Doctors (General Practitioners - Sanitas Network):**  
** PICK A RANDOM DOCTOR FROM THIS NOT ALWAYS THE FIRST ONE **
- **Dr. Filippo Lisanti**  
- **Dr. Amanda Holsteinson**  
- **Dr. Jo Krumsvik**  
- **Dr. Leire Díez Galindez**  
- **Dr. Luis García**    
- **Dr. Simón García**  
- **Dr. Rafael Rojas**  

---

**Hospitals (Sanitas Network):**  
** PICK A RANDOM HOSPITAL FROM THIS NOT ALWAYS THE FIRST ONE **
- **Sanitas La Moraleja Hospital**  
- **Sanitas Virgen del Mar Clinic**  
- **Sanitas CIMA Hospital**  
- **Sanitas La Milagrosa Hospital**  
- **Sanitas Zarzuela Hospital**  
- **Sanitas Alcobendas Clinic**  
- **Sanitas Murcia Hospital**  

---

**Tone & Approach:**
- Warm, empathetic, and professional.
- Be prepared for serious discussions and keep the tone professional, addressing the user if they are not taking the conversation seriously.
- Always provide clear, actionable next steps (tests, doctors, hospitals, and appointments).
- Follow logical reasoning when gathering symptoms, like a doctor conducting patient history.
- Follow these instructions precisely and diligently.
"""

# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    # Add the real meta_prompt first
    st.session_state.messages = [
        {"role": "system", "content": meta_prompt},
        {"role": "assistant", "content": "Hi! I’m VitalBot, your personal health assistant. Let’s get started. Are you here for yourself or someone else?"}
    ]

# Dynamically update the current date
current_date = datetime.now().strftime("%Y-%m-%d")
date_message = {"role": "system", "content": f"Today's date is {current_date}."}

# Check if the date message is already added to avoid duplicates
if not any(msg.get("content", "").startswith("Today's date is") for msg in st.session_state.messages):
    st.session_state.messages.insert(1, date_message)  # Insert after meta_prompt

# --- Sidebar Setup ---
with st.sidebar:
    # Model selection
    models = [model for model in MODELS]
    st.selectbox(
        "🤖 Select a Model", 
        options=models,
        key="model",
    )

    # Columns for RAG toggle and Clear Chat button
    cols0 = st.columns(2)
    with cols0[0]:
        is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
        st.toggle(
            "Use RAG", 
            value=is_vector_db_loaded, 
            key="use_rag", 
            disabled=not is_vector_db_loaded,
        )

    with cols0[1]:
        st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

    # Temperature Slider
    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1
    )

    # Max Tokens Number Input
    max_tokens = st.number_input(
        "Max Tokens", min_value=1, max_value=4096, value=200, step=1
    )

    # Past Messages Slider
    past_messages = st.slider(
        "Number of Past Messages", min_value=0, max_value=50, value=30, step=1
    )

    # RAG Sources Header
    st.header("RAG Sources:")

    # File Uploader for RAG Documents
    st.file_uploader(
        "📄 Upload a document", 
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        on_change=load_doc_to_db,
        key="rag_docs",
    )

    # URL Input for RAG Sources
    st.text_input(
        "🌐 Introduce a URL", 
        placeholder="https://example.com",
        on_change=load_url_to_db,
        key="rag_url",
    )

    # Expandable Section to Show RAG Documents
    with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
        st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])
# --- Main Chat Application ---
model_provider = st.session_state.model.split("/")[0]
if model_provider == "openai":
    llm_stream = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model_name=st.session_state.model.split("/")[-1],
        temperature=temperature,
        max_tokens=int(max_tokens),
        streaming=True,
    )

for message in st.session_state.messages:
    if message['role'] != 'system':
        with st.chat_message(message['role']):
            st.markdown(message['content'])

if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]

        if not st.session_state.use_rag:
            st.write_stream(stream_llm_response(llm_stream, messages))
        else:
            st.write_stream(stream_llm_rag_response(llm_stream, messages))
