from fastapi import FastAPI
from pydantic import BaseModel
from rag_chatbotv2 import retrieve_context, generate_answer, memory

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    """Handles incoming chat requests with memory support."""
    query = request.query
    context = retrieve_context(query)
    answer = generate_answer(query, context)

    # Store conversation in memory
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(answer)

    return {"query": query, "answer": answer}

@app.get("/")
def read_root():
    """Root endpoint for confirming API is live."""
    return {"message": "RAG Chatbot API is running!"}

@app.get("/health")
def health_check():
    """Health check endpoint for deployment monitoring."""
    return {"status": "ok"}
