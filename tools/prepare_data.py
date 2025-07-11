import os
import cv2
import json
import glob
import torch
import clip
import faiss
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# --- Configuration ---
SCENES_DIR = "scenes"
EMBEDDINGS_DIR = "embeddings"
IMAGE_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
IMAGE_FILENAMES_FILE = os.path.join(EMBEDDINGS_DIR, "image_filenames.json")
FAISS_INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "faiss.index")

# --- Preprocessing Functions ---

def find_scenes(video_path):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=45, min_scene_len=30))
    scene_manager.detect_scenes(video, show_progress=True)
    return scene_manager.get_scene_list()

def save_scene_keyframes(video_path, scene_list, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    saved_image_paths = []

    for i, scene in enumerate(tqdm(scene_list, desc="Extracting keyframes")):
        start_frame, end_frame = scene[0].get_frames(), scene[1].get_frames()
        middle_frame = start_frame + (end_frame - start_frame) // 2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        
        if ret:
            output_path = os.path.join(output_dir, f"scene_{i+1:03d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_image_paths.append(output_path)
            
    cap.release()
    return saved_image_paths

def save_scene_videos(video_path, scene_list, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    for i, scene in enumerate(tqdm(scene_list, desc="Exporting scenes")):
        start_frame, end_frame = scene[0].get_frames(), scene[1].get_frames()
        output_path = os.path.join(output_dir, f"scene_{i+1:03d}.mp4")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)
        out.release()
    
    cap.release()

# --- Indexing Functions ---

def create_embeddings(image_paths):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    all_features = []
    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Generating embeddings"):
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            features = model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())

    return np.vstack(all_features)

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

# --- Main Execution ---

def main(args):
    # --- 1. Preprocessing ---
    print("--- Step 1: Video Preprocessing ---")
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        return
        
    scene_list = find_scenes(args.video_path)
    print(f"Found {len(scene_list)} scenes.")
    
    save_scene_keyframes(args.video_path, scene_list, SCENES_DIR)
    save_scene_videos(args.video_path, scene_list, SCENES_DIR)
    print("Keyframes and scene videos saved.")

    # --- 2. Indexing ---
    print("\n--- Step 2: Feature Extraction and Indexing ---")
    if not os.path.exists(EMBEDDINGS_DIR):
        os.makedirs(EMBEDDINGS_DIR)
        
    image_paths = sorted(glob.glob(os.path.join(SCENES_DIR, "*.jpg")))
    if not image_paths:
        print("Error: No keyframes found to index.")
        return

    embeddings = create_embeddings(image_paths)
    np.save(IMAGE_EMBEDDINGS_FILE, embeddings)
    print(f"Embeddings saved to {IMAGE_EMBEDDINGS_FILE}")

    filenames = [os.path.basename(p) for p in image_paths]
    with open(IMAGE_FILENAMES_FILE, 'w') as f:
        json.dump(filenames, f)
    print(f"Filenames saved to {IMAGE_FILENAMES_FILE}")
    
    index = build_faiss_index(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"FAISS index saved to {FAISS_INDEX_FILE}")

    print("\nData preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare video data for scene retrieval.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    args = parser.parse_args()
    main(args) 