# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Config / env ---
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "vector_db")
TEXT_COLLECTION = os.getenv("TEXT_COLLECTION", "texts")

if not MONGODB_URI:
    raise RuntimeError("Set MONGODB_URI in .env before running")

# --- App and DB ---
app = FastAPI(title="MongoDB Vector Search API")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
texts_coll = db[TEXT_COLLECTION]

# --- Load embedding model (blocking on startup) ---
# Note: model loads into memory once when the app starts
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# --- Request / Response schemas ---
class AddTextRequest(BaseModel):
    title: str
    content: str

class AddTextResponse(BaseModel):
    inserted_id: str
    title: str

# --- Helper: convert numpy array to list of floats ---
def _to_list(vector):
    # sentence-transformers encode returns numpy array or list
    return vector.tolist() if hasattr(vector, "tolist") else list(vector)

# --- Endpoint: Health check ---
@app.get("/")
def root():
    return {"message": "MongoDB Vector Search API is running!"}

# --- Endpoint: Add text and embedding ---
@app.post("/add-text", response_model=AddTextResponse)
def add_text(payload: AddTextRequest):
    """
    Accepts title and content, generates embedding, stores document in MongoDB.
    """
    try:
        text = f"{payload.title}\n\n{payload.content}"
        # Generate embedding (returns numpy array)
        embedding = model.encode(text)
        embedding_list = _to_list(embedding)

        # Document to insert
        doc = {
            "title": payload.title,
            "content": payload.content,
            "embedding": embedding_list,
            "model": MODEL_NAME
        }
        res = texts_coll.insert_one(doc)

        return {"inserted_id": str(res.inserted_id), "title": payload.title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
