import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HOME"] = "/tmp"
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain.memory import ConversationBufferMemory

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("changi-jewel-index")
os.environ["HF_HOME"] = "/tmp/huggingface"  
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def retrieve_context(query, top_k=3):
    """Retrieve top-k relevant documents from Pinecone."""
    query_embedding = embedder.encode([query]).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    contexts = [m["metadata"].get("text", "") for m in results["matches"]]
    return "\n\n".join(contexts)

def generate_answer(query, context):
    """Use GPT-4o-mini with memory and retrieved context."""
    # Get chat history from memory
    chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in memory.load_memory_variables({})["chat_history"]])

    prompt = f"""
                 You are an AI chat assistant with knowledge of Changi Airport Group and Changi Jewel Airport.
                 STRICTLY ONLY use the context below to answer the user's question accurately.
                 DONOT ANSWER ANYTHING outside the context provided.
                 When the user asks about information not in the context, respond with "Sorry, I can only provide 
                 information regarding Changi Airport Group and Jewel Changi Airport."

                Chat History:
                {chat_history}

                Context:
                {context}

                Question: {query}

                Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    answer = response.choices[0].message.content

    # Save to memory
    memory.save_context({"input": query}, {"output": answer})

    return answer

def chat():
    print("\nContextual RAG Chatbot ready! Type 'bye' to quit.")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() == "bye":
            break
        context = retrieve_context(query)
        answer = generate_answer(query, context)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    chat()
