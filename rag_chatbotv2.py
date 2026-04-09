"""
Changi RAG Chatbot Backend
Uses LangChain 0.3.x with:
  - LCEL (LangChain Expression Language) chains
  - RunnableWithMessageHistory for persistent, per-session memory
  - langchain-openai for the LLM
  - Pinecone + SentenceTransformers for retrieval
"""

import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# ---------------------------------------------------------------------------
# Environment & Clients
# ---------------------------------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise EnvironmentError(
        "Missing API keys. Ensure PINECONE_API_KEY and OPENAI_API_KEY are set in .env"
    )

# Pinecone retriever
pc      = Pinecone(api_key=PINECONE_API_KEY)
index   = pc.Index("changi-jewel-index")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# LLM
llm = ChatOpenAI(
    model="gpt-4.1-mini-2025-04-14",
    temperature=0,
    api_key=OPENAI_API_KEY,
)

# ---------------------------------------------------------------------------
# In-memory store keyed by session_id (populated and owned by streamlit_app)
# ---------------------------------------------------------------------------
session_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """Return (or create) a ChatMessageHistory for the given session."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


# ---------------------------------------------------------------------------
# Retrieval helper
# ---------------------------------------------------------------------------
def retrieve_context(query: str, top_k: int = 4) -> str:
    """Embed the query and pull the top-k passages from Pinecone."""
    vector   = embedder.encode([query]).tolist()
    results  = index.query(vector=vector, top_k=top_k, include_metadata=True)
    passages = [m["metadata"].get("text", "") for m in results["matches"]]
    return "\n\n".join(passages)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a friendly AI assistant for Changi Airport Group and Jewel Changi Airport.

GUIDELINES:
1. For greetings, small talk, or pleasantries (e.g. "hi", "hello", "thanks"), respond naturally and warmly without referencing the context.
2. For factual questions, use ONLY the provided context to answer. Answer from the context provided ONLY.
3. If a factual question is asked but the context does not contain a relevant answer, respond with:
   "Sorry, I can only provide information regarding Changi Airport Group and Jewel Changi Airport."
4. Keep answers concise and helpful.

Context (use this for factual questions only):
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# ---------------------------------------------------------------------------
# Chain  (context is injected before calling the chain)
# ---------------------------------------------------------------------------
_base_chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    _base_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


# ---------------------------------------------------------------------------
# Public API used by streamlit_app.py
# ---------------------------------------------------------------------------
# Short messages that are clearly greetings / small-talk — skip Pinecone lookup
_GREETING_TRIGGERS = {
    "hi", "hello", "hey", "hiya", "howdy", "greetings",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "bye", "goodbye", "ok", "okay", "sure",
}


def _is_greeting(text: str) -> bool:
    return text.strip().lower().rstrip("!") in _GREETING_TRIGGERS


def chat(question: str, session_id: str = "default") -> str:
    """
    Full RAG + memory pipeline.

    Args:
        question:   The user's message.
        session_id: Unique identifier for the conversation thread.
                    Pass a different ID to start a fresh memory context.

    Returns:
        The assistant's reply as a string.
    """
    # Skip vector search for simple greetings to avoid empty-context fallback
    context = "" if _is_greeting(question) else retrieve_context(question)

    response = chain_with_history.invoke(
        {"question": question, "context": context},
        config={"configurable": {"session_id": session_id}},
    )
    return response.content


def clear_session(session_id: str = "default") -> None:
    """Wipe the chat history for the given session."""
    if session_id in session_store:
        session_store[session_id].clear()


def get_history(session_id: str = "default") -> list[dict]:
    """
    Return chat history as a simple list of dicts for display.
    Format: [{"role": "human"|"ai", "content": "..."}]
    """
    history = get_session_history(session_id).messages
    result  = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            result.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "ai", "content": msg.content})
    return result


# ---------------------------------------------------------------------------
# CLI mode (for quick testing without Streamlit)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nChangi RAG Chatbot  |  type 'bye' to quit, 'clear' to reset memory\n")
    sid = "cli-session"
    while True:
        q = input("You: ").strip()
        if q.lower() == "bye":
            break
        if q.lower() == "clear":
            clear_session(sid)
            print("[Memory cleared]\n")
            continue
        answer = chat(q, session_id=sid)
        print(f"\nAssistant: {answer}\n")
