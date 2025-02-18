# app.py
from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
import fitz  # PyMuPDF
from google.cloud import storage
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# Initialize Flask app
app = Flask(__name__)

# Initialize clients
storage_client = storage.Client()
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
generation_model = GenerativeModel("gemini-1.5-pro-002")

BUCKET_NAME = "research_gcp"
DOCS_PREFIX = "documents/"
CACHE_BLOB_NAME = "cache/embeddings_cache.pkl"
MAX_TEXT_LENGTH = 2000  # Adjust as needed

def split_text_into_chunks(text, max_length=MAX_TEXT_LENGTH):
    lines = text.splitlines()
    chunks = []
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += "\n" + line if current_chunk else line
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def load_cache_from_gcs():
    """Load the embeddings cache from Cloud Storage if it exists."""
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(CACHE_BLOB_NAME)
        if blob.exists():
            cache_bytes = blob.download_as_bytes()
            embeddings = pickle.loads(cache_bytes)
            print("Loaded embeddings from GCS cache.")
            return embeddings
        else:
            print("Cache blob does not exist in GCS.")
            return None
    except Exception as e:
        print("Error loading cache from GCS:", e)
        return None

def save_cache_to_gcs(embeddings):
    """Save the embeddings cache to Cloud Storage."""
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(CACHE_BLOB_NAME)
        cache_bytes = pickle.dumps(embeddings)
        blob.upload_from_string(cache_bytes)
        print("Embeddings cached to GCS successfully.")
    except Exception as e:
        print("Error saving cache to GCS:", e)

def retrieve_and_process_pdfs():
    # Attempt to load from GCS cache first
    embeddings = load_cache_from_gcs()
    if embeddings is not None:
        return embeddings

    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=DOCS_PREFIX)
    embeddings = []

    for blob in blobs:
        if blob.name.endswith(".pdf"):
            try:
                pdf_bytes = blob.download_as_bytes()
                doc = fitz.open("pdf", pdf_bytes)
                full_text = "\n".join([page.get_text() for page in doc])
                text_chunks = split_text_into_chunks(full_text)
                chunk_embeddings = []
                for chunk in text_chunks:
                    result = embedding_model.get_embeddings([chunk])
                    chunk_embedding = result[0].values
                    chunk_embeddings.append(np.array(chunk_embedding))
                if chunk_embeddings:
                    avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
                else:
                    avg_embedding = embedding_model.get_embeddings([full_text])[0].values

                embeddings.append({
                    "name": blob.name,
                    "embedding": avg_embedding,
                    "text": full_text
                })
            except Exception as e:
                print(f"Error processing {blob.name}: {e}")

    save_cache_to_gcs(embeddings)
    return embeddings

def query_and_generate_answer(query, embeddings):
    if not embeddings:
        return "No documents available for generating an answer."
    try:
        query_embedding = embedding_model.get_embeddings([query])[0].values
        query_embedding = np.array(query_embedding)
    except Exception as e:
        return f"Error generating query embedding: {e}"

    def cosine_similarity(vec1, vec2):
        vec1, vec2 = np.array(vec1), np.array(vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norm_product == 0:
            return 0
        return np.dot(vec1, vec2) / norm_product

    similarities = []
    for doc in embeddings:
        try:
            sim = cosine_similarity(query_embedding, doc["embedding"])
            similarities.append((sim, doc["text"]))
        except Exception as e:
            print(f"Error computing similarity for {doc['name']}: {e}")

    if not similarities:
        return "No valid embeddings to compare."

    best_match = max(similarities, key=lambda x: x[0])
    best_context = best_match[1]

    if len(best_context) > MAX_TEXT_LENGTH:
        best_context = best_context[:MAX_TEXT_LENGTH]
        print("Warning: The best context was trimmed due to length limits.")

    prompt = f"Using the following context:\n{best_context}\nAnswer the user query: {query}"
    try:
        response = generation_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {e}"

# Pre-load and cache embeddings at startup (optional)
EMBEDDINGS = retrieve_and_process_pdfs()
print(EMBEDDINGS)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request"}), 400
    query = data["query"]
    answer = query_and_generate_answer(query, EMBEDDINGS)
    return jsonify({"answer": answer})

@app.route("/", methods=["GET"])
def index():
    return "API is up and running!"

if __name__ == "__main__":
    # Use the PORT environment variable if set, defaulting to 8080
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
