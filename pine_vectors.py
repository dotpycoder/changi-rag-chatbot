import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, VectorType
from sentence_transformers import SentenceTransformer

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "changi-jewel-index"
if not pc.has_index(index_name):
    config = pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
        vector_type=VectorType.DENSE
    )
else:
    config = pc.describe_index(index_name)

index = pc.Index(host=config.host)  # connect to the index

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=500, overlap=150):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks

# Clear all existing vectors in the index
print("\nClearing existing vectors...\n")
index.delete(delete_all=True)
stats = index.describe_index_stats()
print(f"Vectors after clearing: {stats['total_vector_count']}")
print("\nIndex cleared. Re-ingesting data...\n")

for fname in os.listdir("data"):
    text = open(os.path.join("data", fname), encoding="utf-8").read()
    chunks = chunk_text(text)
    embeddings = model.encode(chunks).tolist()
    ids = [f"{fname}_{i}" for i in range(len(chunks))]
    vectors = [
        {"id": ids[i], "values": embeddings[i], "metadata": {"source": fname, "text": chunks[i]}}
        for i in range(len(chunks))
    ]
    index.upsert(vectors=vectors)

print("✅ Ingestion to Pinecone complete.")
