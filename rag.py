import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI

# Load your JSON file
with open("scraped_chunks.json", "r") as f:
    data = json.load(f)

# Create a list of string chunks (e.g., key + value)
chunks = [f"{item['key']}: {item['value']}" for item in data]


model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, show_progress_bar=True)

embedding_array = np.array(embeddings).astype("float32")
index = faiss.IndexFlatL2(embedding_array.shape[1])
index.add(embedding_array)

# Map index to chunk text
id_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}


def retrieve_chunks(query, top_k=5):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [id_to_chunk[i] for i in indices[0]]

API = ""

client = OpenAI(api_key=API)

def generate_answer(prompt):
    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful assistant for WooCommerce GST Plugin users."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

print(generate_answer("Guide me through the process of setting up the plug in"))
