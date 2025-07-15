import os
import cv2
import json
import glob
import torch
import clip
import faiss
import numpy as np
import argparse
import shutil
from PIL import Image
from tqdm import tqdm
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# --- Configuration ---
SCENES_DIR = "scenes"
EMBEDDINGS_DIR = "embeddings"
IMAGE_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
IMAGE_FILENAMES_FILE = os.path.join(EMBEDDINGS_DIR, "image_filenames.json")
VIDEO_SOURCES_FILE = os.path.join(EMBEDDINGS_DIR, "video_sources.json")
FAISS_INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "faiss.index")
TIMECODES_FILE = os.path.join(EMBEDDINGS_DIR, "scene_timecodes.json")
SUPPORTED_VIDEO_EXTENSIONS = ['*.mp4', '*.mkv', '*.avi', '*.mov']

# --- Core Processing Functions ---

def find_scenes(video_path):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=45, min_scene_len=30))
    scene_manager.detect_scenes(video, show_progress=True)
    return scene_manager.get_scene_list()

def save_scene_keyframes(video_path, scene_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    for i, scene in enumerate(tqdm(scene_list, desc="Extracting keyframes")):
        start_frame, end_frame = scene[0].get_frames(), scene[1].get_frames()
        middle_frame = start_frame + (end_frame - start_frame) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, f"scene_{i+1:03d}.jpg"), frame)
    cap.release()

def create_embeddings(image_paths):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for CLIP embedding")
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

# --- Main Logic for a Single Video ---

def process_single_video(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}, skipping.")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_scenes_dir = os.path.join(SCENES_DIR, video_name)
    print(f"\n--- Processing video: {video_name} ---")

    # 1. Preprocessing
    scene_list = find_scenes(video_path)
    print(f"Found {len(scene_list)} scenes.")
    save_scene_keyframes(video_path, scene_list, video_scenes_dir)
    print(f"Keyframes saved to {video_scenes_dir}")

    # 2. Load existing data
    all_embeddings = []
    all_filenames = []
    all_sources = []
    all_timecodes = {}
    if os.path.exists(IMAGE_EMBEDDINGS_FILE):
        all_embeddings = list(np.load(IMAGE_EMBEDDINGS_FILE))
        with open(IMAGE_FILENAMES_FILE, 'r') as f: all_filenames = json.load(f)
        if os.path.exists(VIDEO_SOURCES_FILE):
            with open(VIDEO_SOURCES_FILE, 'r') as f: all_sources = json.load(f)
        with open(TIMECODES_FILE, 'r') as f: all_timecodes = json.load(f)
        print("Loaded existing library data.")

    # 3. Clean up old entries for this video
    indices_to_keep = [i for i, f in enumerate(all_filenames) if not f.startswith(video_name + os.path.sep)]
    if len(indices_to_keep) < len(all_filenames):
        all_embeddings = [all_embeddings[i] for i in indices_to_keep]
        all_filenames = [all_filenames[i] for i in indices_to_keep]
        if all_sources:
             all_sources = [all_sources[i] for i in indices_to_keep]
        if video_name in all_timecodes: del all_timecodes[video_name]
        print(f"Removed existing entries for '{video_name}'.")

    # 4. Generate and append new data
    new_image_paths = sorted(glob.glob(os.path.join(video_scenes_dir, "*.jpg")))
    new_embeddings = create_embeddings(new_image_paths)
    new_filenames = [os.path.join(video_name, os.path.basename(p)) for p in new_image_paths]
    new_sources = [os.path.basename(video_path)] * len(new_filenames)
    new_timecodes = {f"scene_{(i+1):03d}": (s[0].get_seconds(), s[1].get_seconds()) for i, s in enumerate(scene_list)}

    # 5. Combine and save
    final_embeddings = all_embeddings + list(new_embeddings)
    final_filenames = all_filenames + new_filenames
    final_sources = all_sources + new_sources
    all_timecodes[video_name] = new_timecodes
    
    if not final_embeddings:
        print("No embeddings to save. Exiting.")
        return

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    np.save(IMAGE_EMBEDDINGS_FILE, np.vstack(final_embeddings))
    with open(IMAGE_FILENAMES_FILE, 'w') as f: json.dump(final_filenames, f, indent=4)
    with open(VIDEO_SOURCES_FILE, 'w') as f: json.dump(final_sources, f, indent=4)
    with open(TIMECODES_FILE, 'w') as f: json.dump(all_timecodes, f, indent=4)
    print("Combined data saved.")
    
    # 6. Re-build FAISS index
    index = build_faiss_index(np.vstack(final_embeddings))
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"FAISS index rebuilt and saved.")

# --- Cleaning Functions ---

def clean_all_data():
    print("--- Cleaning all data ---")
    if os.path.exists(SCENES_DIR):
        shutil.rmtree(SCENES_DIR)
        print(f"Removed directory: {SCENES_DIR}")
    if os.path.exists(EMBEDDINGS_DIR):
        shutil.rmtree(EMBEDDINGS_DIR)
        print(f"Removed directory: {EMBEDDINGS_DIR}")
    print("All data cleaned.")

def clean_single_video_data(video_path):
    print(f"--- Cleaning data for: {video_path} ---")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 1. Remove scenes folder
    video_scenes_dir = os.path.join(SCENES_DIR, video_name)
    if os.path.exists(video_scenes_dir):
        shutil.rmtree(video_scenes_dir)
        print(f"Removed scenes directory: {video_scenes_dir}")

    # 2. Remove from embedding files if they exist
    if not os.path.exists(EMBEDDINGS_DIR):
        print("Embeddings directory not found, nothing to clean.")
        return

    all_embeddings, all_filenames, all_timecodes, all_sources = [], [], {}, []
    if os.path.exists(IMAGE_EMBEDDINGS_FILE):
        all_embeddings = list(np.load(IMAGE_EMBEDDINGS_FILE))
        with open(IMAGE_FILENAMES_FILE, 'r') as f: all_filenames = json.load(f)
        with open(TIMECODES_FILE, 'r') as f: all_timecodes = json.load(f)
        if os.path.exists(VIDEO_SOURCES_FILE):
            with open(VIDEO_SOURCES_FILE, 'r') as f: all_sources = json.load(f)

    indices_to_keep = [i for i, f in enumerate(all_filenames) if not f.startswith(video_name + os.path.sep)]
    
    if len(indices_to_keep) == len(all_filenames):
        print(f"No data found for '{video_name}' in the index. Nothing to clean from embeddings.")
        return

    final_embeddings = [all_embeddings[i] for i in indices_to_keep]
    final_filenames = [all_filenames[i] for i in indices_to_keep]
    if all_sources:
        final_sources = [all_sources[i] for i in indices_to_keep]
    else:
        final_sources = []
    if video_name in all_timecodes: del all_timecodes[video_name]

    print(f"Removed {len(all_filenames) - len(final_filenames)} entries from index.")

    if not final_filenames:
        # If no files are left, clean up everything
        clean_all_data()
    else:
        # 3. Save updated files and rebuild index
        np.save(IMAGE_EMBEDDINGS_FILE, np.vstack(final_embeddings))
        with open(IMAGE_FILENAMES_FILE, 'w') as f: json.dump(final_filenames, f, indent=4)
        with open(TIMECODES_FILE, 'w') as f: json.dump(all_timecodes, f, indent=4)
        if final_sources or os.path.exists(VIDEO_SOURCES_FILE):
            with open(VIDEO_SOURCES_FILE, 'w') as f: json.dump(final_sources, f, indent=4)
        index = build_faiss_index(np.vstack(final_embeddings))
        faiss.write_index(index, FAISS_INDEX_FILE)
        print("Embeddings and index rebuilt after cleaning.")

# --- Main Dispatcher ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage the video scene retrieval library.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to a single video file to add or update.")
    group.add_argument("--folder", type=str, help="Path to a folder with videos to add or update.")
    group.add_argument("--clean-video", type=str, help="Path to a single video file to remove from the library.")
    group.add_argument("--clean-all", action='store_true', help="Remove all indexed data and scenes.")
    
    args = parser.parse_args()

    if args.video:
        process_single_video(args.video)
    elif args.folder:
        print(f"--- Processing all videos in folder: {args.folder} ---")
        
        # Check for already processed videos to avoid redundant work
        processed_videos = set()
        if os.path.exists(TIMECODES_FILE):
            with open(TIMECODES_FILE, 'r') as f:
                timecodes_data = json.load(f)
                processed_videos = set(timecodes_data.keys())
            print(f"Found {len(processed_videos)} videos already in the library.")

        video_paths = []
        for ext in SUPPORTED_VIDEO_EXTENSIONS:
            search_pattern = os.path.join(glob.escape(args.folder), '**', ext)
            video_paths.extend(glob.glob(search_pattern, recursive=True))
        
        if not video_paths:
            print("No video files found in the specified folder.")
        else:
            new_videos_found = 0
            for video_path in video_paths:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                if video_name in processed_videos:
                    print(f"Skipping '{video_path}' as it is already processed.")
                    continue
                
                new_videos_found += 1
                process_single_video(video_path)

            if new_videos_found == 0:
                print("No new videos to process.")

    elif args.clean_video:
        clean_single_video_data(args.clean_video)
    elif args.clean_all:
        clean_all_data()

    print("\nOperation complete.") 