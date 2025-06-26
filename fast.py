import os
import json
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# === Load API Key ===
API_KEY = "YOUR_OPENAI_API_KEY"  # Replace with your actual OpenAI API key
client = OpenAI(api_key=API_KEY)

# ── 2. Load pre-chunked RAG data ───────────────────────────────────────────
try:
    with open("scraped_chunks.json", "r") as f:
        data = json.load(f)              # expects a list of {"key": ..., "value": ...}
    texts = [f"{chunk['key']}: {chunk['value']}" for chunk in data]
except FileNotFoundError:
    raise FileNotFoundError("scraped_chunks.json not found in current directory")
except json.JSONDecodeError:
    raise ValueError("Invalid JSON format in scraped_chunks.json")

# Sentence-Transformers → 384-dim embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")

print("Chunks loaded:", len(texts))
print("Embeddings shape:", embeddings.shape)   # (N, 384)

# ── 3. Build FAISS index ───────────────────────────────────────────────────
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# ── 4. FastAPI app with CORS ───────────────────────────────────────────────
app = FastAPI(title="RAG API", description="Retrieval Augmented Generation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # ← swap "*" for your site in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryInput(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# ── 5. Chat endpoint ───────────────────────────────────────────────────────
@app.post("/ask", response_model=QueryResponse)
async def ask_question(input: QueryInput):
    try:
        # Validate input
        if not input.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Vectorize the incoming query
        query_emb = model.encode([input.query], convert_to_numpy=True).astype("float32")
        distances, idxs = index.search(query_emb, k=5)
        
        # Build context from top-k chunks
        context = "\n\n".join(texts[i] for i in idxs[0])
        
        prompt = (
            "Use the context below to answer the question. If the context doesn't contain "
            "enough information to answer the question, please say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{input.query}"
        )
        
        # Note: Remove 'await' - OpenAI client is synchronous by default
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content.strip()
        return QueryResponse(answer=answer)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# ── 6. Health check endpoints ──────────────────────────────────────────────
@app.get("/")
async def read_root():
    return {
        "message": "RAG API is running", 
        "total_chunks": len(texts),
        "embedding_dim": embeddings.shape[1]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "chunks_loaded": len(texts)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)