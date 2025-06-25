import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# === Load API Key ===
API_KEY = ""
client = OpenAI(api_key=API_KEY)

# === Load your RAG chunks ===
with open("scraped_chunks.json", "r") as f:
    data = json.load(f)

texts = [f"{chunk['key']}: {chunk['value']}" for chunk in data]
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")

# === Setup FAISS ===
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# === FastAPI Setup ===
app = FastAPI()

class QueryInput(BaseModel):
    query: str

@app.post("/ask")
def ask_question(input: QueryInput):
    # Vectorize query
    query_embedding = model.encode([input.query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, 5)

    # Build context
    context = "\n\n".join([texts[i] for i in indices[0]])

    # Prepare prompt
    prompt = f"Use the context below to answer the question:\n\nContext:\n{context}\n\nQuestion:\n{input.query}"

    # Ask OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    answer = response.choices[0].message.content

    return {"answer": answer}
