# RAG Chatbot for Jewel Changi Airport

A Retrieval-Augmented Generation (RAG) chatbot that integrates **LLMs** with **Pinecone vector database** for intelligent, context-aware responses. Deployed on **Hugging Face Spaces** with a **Gradio-based UI**.

## Features
- **Web Crawled 2 websites:** Scraped the 2 websites upto **max_depth=3**, which means it has scraped in from the <href> links on both the websites. 
- **Contextual Q&A:** Uses RAG to retrieve the most relevant information without **Hallucinating**.
- **FastAPI Backend:** Provides REST endpoints for chatbot interaction.
- **Gradio UI:** Simple and interactive frontend for users.
- **Scalable Deployment:** Hosted on Hugging Face Spaces for easy access and collaboration.
the chatbot is live here.
