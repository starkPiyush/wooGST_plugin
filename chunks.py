import json
import uuid
import os

def chunk_scraped_json(file_path, output_path='scraped_chunks.json'):
    """
    Reads key-value pairs from a JSON file and writes structured chunks
    to a new JSON file. Each chunk contains 'id', 'key', and 'value'.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = []
    for key, value in data.items():
        chunk = {
            "id": str(uuid.uuid4()),
            "key": key,
            "value": value
        }
        chunks.append(chunk)

    # Save to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Chunks saved to '{output_path}'")
    return chunks


chunks = chunk_scraped_json('scrapped_data.json')
