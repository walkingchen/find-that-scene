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
VIDEO_SOURCES_FILE = os.path.join(EMBEDDINGS_DIR, "video_sources.json")
FAISS_INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "faiss.index")
TIMECODES_FILE = os.path.join(EMBEDDINGS_DIR, "scene_timecodes.json")
METADATA_FILE = os.path.join(EMBEDDINGS_DIR, "metadata.json")
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
scene_timecodes = None
video_sources = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Models ---
class SearchResult(BaseModel):
    scene_id: int
    keyframe_filename: str
    keyframe_path: str
    source_video_filename: str
    start_time: float
    end_time: float
    score: float
    video_path: str

# --- Lifespan Events ---
@app.on_event("startup")
def startup_event():
    global clip_model, clip_preprocess, faiss_index, image_filenames, scene_timecodes, video_sources
    
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

    # Load scene timecodes
    if not os.path.exists(TIMECODES_FILE):
        raise HTTPException(status_code=500, detail="Scene timecodes not found.")
    with open(TIMECODES_FILE, 'r') as f:
        scene_timecodes = json.load(f)
    print("Scene timecodes loaded.")
    
    # Load video sources
    if not os.path.exists(VIDEO_SOURCES_FILE):
        raise HTTPException(status_code=500, detail="Video sources file not found.")
    with open(VIDEO_SOURCES_FILE, 'r') as f:
        video_sources = json.load(f)
    print("Video sources loaded.")

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
        
        # Filename now includes the video name (e.g., "video_name/scene_001.jpg")
        full_keyframe_path = image_filenames[idx]
        video_name, keyframe_filename = os.path.split(full_keyframe_path)
        scene_key = os.path.splitext(keyframe_filename)[0]
        scene_num = int(scene_key.replace("scene_", ""))

        # Look up timecodes in the nested dictionary
        start_time, end_time = scene_timecodes.get(video_name, {}).get(scene_key, (0, 0))
        
        source_filename = video_sources[idx]
        video_path = os.path.join("video", source_filename)

        result = SearchResult(
            scene_id=scene_num,
            keyframe_filename=keyframe_filename,
            keyframe_path=os.path.join(SCENES_DIR, full_keyframe_path),
            source_video_filename=source_filename,
            start_time=start_time,
            end_time=end_time,
            score=float(distances[0][i]),
            video_path=video_path
        )
        results.append(result)
        
    return results

# --- To run the app: uvicorn main:app --reload ---
