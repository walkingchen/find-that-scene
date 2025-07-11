import os
import json
import torch
import clip
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Configuration ---
EMBEDDINGS_DIR = "embeddings"
SCENES_DIR = "scenes"
IMAGE_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
IMAGE_FILENAMES_FILE = os.path.join(EMBEDDINGS_DIR, "image_filenames.json")
FAISS_INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "faiss.index")
TOP_K = 3  # Number of results to return

# --- FastAPI App ---
app = FastAPI()

# Allow CORS for the frontend to access video files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your Gradio app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Globals ---
clip_model = None
clip_preprocess = None
faiss_index = None
image_filenames = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Models ---
class SearchResult(BaseModel):
    scene_id: int
    filename: str
    video_url: str
    score: float

# --- Lifespan Events ---
@app.on_event("startup")
def startup_event():
    global clip_model, clip_preprocess, faiss_index, image_filenames
    
    print("Loading models and data...")
    
    # Load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    print("CLIP model loaded.")

    # Load FAISS index
    if not os.path.exists(FAISS_INDEX_FILE):
        raise HTTPException(status_code=500, detail="FAISS index not found.")
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    print("FAISS index loaded.")

    # Load image filenames
    if not os.path.exists(IMAGE_FILENAMES_FILE):
        raise HTTPException(status_code=500, detail="Image filenames not found.")
    with open(IMAGE_FILENAMES_FILE, 'r') as f:
        image_filenames = json.load(f)
    print("Image filenames loaded.")
    
    print("Startup complete.")

# --- API Endpoints ---
@app.get("/search", response_model=list[SearchResult])
def search(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is missing.")

    if clip_model is None or faiss_index is None:
        raise HTTPException(status_code=503, detail="Server is not ready, models are loading.")

    with torch.no_grad():
        text_inputs = clip.tokenize([query]).to(device)
        text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    query_vector = text_features.cpu().numpy().astype('float32')
    
    # Search FAISS index
    distances, indices = faiss_index.search(query_vector, TOP_K)
    
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        filename = image_filenames[idx]
        scene_num = int(filename.replace("scene_", "").replace(".jpg", ""))
        video_filename = f"scene_{scene_num:03d}.mp4"
        
        result = SearchResult(
            scene_id=scene_num,
            filename=filename,
            video_url=f"/{SCENES_DIR}/{video_filename}",
            score=float(distances[0][i])
        )
        results.append(result)
        
    return results

# Mount static files directory
app.mount(f"/{SCENES_DIR}", StaticFiles(directory=SCENES_DIR), name="scenes")

# --- To run the app: uvicorn main:app --reload ---
