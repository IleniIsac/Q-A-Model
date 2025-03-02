from fastapi import FastAPI, Query, HTTPException
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI()

# Load Sentence Transformer model
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Load dataset
dataset_path = "ai_knowledge_base.txt"
try:
    with open(dataset_path, "r") as f:
        documents = [doc.strip() for doc in f.readlines() if doc.strip()]
    
    if not documents:
        raise HTTPException(status_code=500, detail="Dataset is empty! Ensure 'ai_knowledge_base.txt' has valid content.")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail=f"Dataset file '{dataset_path}' not found!")

# Convert documents into vectors
try:
    doc_embeddings = np.array([embedder.encode(doc) for doc in documents], dtype=np.float32)

    # Initialize FAISS index
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error initializing FAISS: {str(e)}")

def search_faiss(query: str, top_k: int = 3):
    """Searches FAISS for the most relevant documents based on query"""
    try:
        query_embedding = np.array([embedder.encode(query)], dtype=np.float32)
        distances, indices = index.search(query_embedding, top_k)
        
        # Ensure valid indices
        results = [documents[i] for i in indices[0] if i < len(documents)]
        return results if results else ["No relevant results found."]
    except Exception as e:
        return [f"Error during search: {str(e)}"]

# API Endpoint
@app.get("/ask")
def ask_question(q: str = Query(..., description="Enter your question")):
    retrieved_context = search_faiss(q, top_k=3)

    return {
        "query": q,
        "retrieved_context": retrieved_context
    }
