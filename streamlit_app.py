"""
Changi RAG Chatbot — Streamlit UI
Clean interface that talks to the modernised backend in rag_chatbotv2.py.
"""

import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from rag_chatbotv2 import chat, clear_session

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Changi Jewel AI Assistant",
    page_icon="✈️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Minimal premium styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #f0f0f0;
    }

    /* Title */
    .hero-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .hero-sub {
        text-align: center;
        color: #aacfd0;
        margin-bottom: 24px;
        font-style: italic;
        font-size: 0.9rem;
    }

    /* Chat bubbles */
    .stChatMessage {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 16px !important;
        padding: 8px !important;
        margin-bottom: 10px !important;
    }

    /* Input box */
    .stChatInput textarea {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,215,0,0.4) !important;
        border-radius: 12px !important;
        color: #fff !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(0,0,0,0.35) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ✈️ Changi AI Assistant")
    st.markdown("---")
    st.markdown(
        "Powered by **gpt-4.1-mini** + **Pinecone RAG**.\n\n"
        "Ask me anything about Changi Airport or Jewel Changi!"
    )
    st.markdown("---")
    if st.button("🗑️  Clear Chat History"):
        st.session_state.messages = []
        clear_session(st.session_state.session_id)
        st.rerun()

# ---------------------------------------------------------------------------
# Session initialisation
# ---------------------------------------------------------------------------
# Use a unique session_id so each browser tab gets isolated memory
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">Changi Jewel Airport Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Experience the world\'s best airport — ask me anything.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "✈️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# User input → RAG pipeline → response
# ---------------------------------------------------------------------------
if user_input := st.chat_input("Help me find the Singapore Airlines terminal..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # Generate response with full RAG + memory
    with st.chat_message("assistant", avatar="✈️"):
        with st.spinner("Thinking..."):
            try:
                response = chat(
                    question=user_input,
                    session_id=st.session_state.session_id,
                )
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as exc:
                err = f"⚠️ Error: {exc}"
                st.error(err)
                st.info("Check that **OPENAI_API_KEY** and **PINECONE_API_KEY** are set in `.env`.")
