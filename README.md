<details>
<summary><strong>:cn: 中文 (Chinese)</strong></summary>

# 文本驱动的视频镜头检索系统

这是一个 MVP（最小可行产品）项目，旨在通过自然语言描述来检索视频中的特定镜头。

## 功能
- **视频处理**: 自动将视频切分成不同的镜头。
- **特征提取**: 使用 OpenAI 的 CLIP 模型为每个镜头提取语义特征向量。
- **快速检索**: 利用 FAISS 为所有镜头建立向量索引，实现快速相似性搜索。
- **Web 界面**: 提供一个简单的 Gradio Web UI，用于输入文本查询并展示、播放匹配的视频片段。

## 技术栈
- **后端**: FastAPI
- **前端**: Gradio
- **模型**: OpenAI CLIP (ViT-B/32)
- **视频处理**: PySceneDetect, FFmpeg
- **向量检索**: FAISS

## 如何运行

### 1. 环境准备
克隆项目后，首先创建并激活 Python 虚拟环境：
```bash
python3 -m venv .venv
source .venv/bin/activate
```

接着，安装所需的依赖：
```bash
pip install -r requirements.txt
```

### 2. 准备视频
将你的视频文件（例如 `my_video.mp4`）放入项目根目录下的 `video/` 文件夹中。你可以放置多个视频。

### 3. 数据预处理和索引
`tools/prepare_data.py` 是一个功能强大的数据管理脚本，你可以用它来构建和维护你的视频索引库。

#### 添加或更新视频
你可以添加单个视频，也可以批量添加一个文件夹内的所有视频。此脚本支持增量更新，会自动跳过已处理的视频。
```bash
# 激活环境并设置变量
source .venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE

# (选项 A) 添加或更新单个视频
python tools/prepare_data.py --video video/demo.mp4

# (选项 B) 批量添加或更新一个文件夹中的所有视频 (推荐)
python tools/prepare_data.py --folder video/
```
这个过程会为每个视频在 `scenes/` 目录下创建独立的文件夹存放关键帧，并更新位于 `embeddings/` 下的统一索引文件。

#### 清理数据
你也可以使用此脚本来清理数据。
```bash
# (选项 C) 清理关于单个视频的所有数据
python tools/prepare_data.py --clean-video video/demo.mp4

# (选项 D) 清理所有已生成的场景和索引数据，完全重置
python tools/prepare_data.py --clean-all
```

### 4. 启动服务
首先，启动 FastAPI 后端服务：
```bash
# 同样需要设置环境变量
export KMP_DUPLICATE_LIB_OK=TRUE

uvicorn main:app --host 0.0.0.0 --port 8000
```

然后，在另一个终端窗口中，启动 Gradio 前端 UI：
```bash
# 激活虚拟环境并设置环境变量
source .venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE

# 启动 UI
python ui.py
```

### 5. 开始使用
打开浏览器，访问 Gradio 提供的本地 URL (通常是 `http://127.0.0.1:7860`)。现在，搜索将会在所有你已处理过的视频中进行。

</details>

<br>

<details open>
<summary><strong>:us: English</strong></summary>

# Text-Driven Video Scene Retrieval System

This is an MVP (Minimum Viable Product) project designed to retrieve specific scenes from videos using natural language descriptions.

## Features
- **Video Processing**: Automatically splits videos into different scenes.
- **Feature Extraction**: Uses OpenAI's CLIP model to extract semantic feature vectors for each scene.
- **Fast Retrieval**: Builds a vector index for all scenes using FAISS for rapid similarity search.
- **Web Interface**: Provides a simple Gradio Web UI for entering text queries and displaying/playing matching video clips.

## Tech Stack
- **Backend**: FastAPI
- **Frontend**: Gradio
- **Model**: OpenAI CLIP (ViT-B/32)
- **Video Processing**: PySceneDetect, FFmpeg
- **Vector Search**: FAISS

## How to Run

### 1. Environment Setup
After cloning the project, first create and activate a Python virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Next, install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare Videos
Place your video files (e.g., `my_video.mp4`) into the `video/` folder in the project's root directory. You can place multiple videos here.

### 3. Data Preprocessing and Indexing
`tools/prepare_data.py` is a powerful data management script you can use to build and maintain your video index library.

#### Add or Update Videos
You can add a single video or batch-add all videos within a folder. This script supports incremental updates and will automatically skip already processed videos.
```bash
# Activate the environment and set the environment variable
source .venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE

# (Option A) Add or update a single video
python tools/prepare_data.py --video video/demo.mp4

# (Option B) Batch add or update all videos in a folder (Recommended)
python tools/prepare_data.py --folder video/
```
This process creates a separate folder for each video's keyframes under `scenes/` and updates the unified index files located in `embeddings/`.

#### Clean Data
You can also use this script to clean up data.
```bash
# (Option C) Clean all data related to a single video
python tools/prepare_data.py --clean-video video/demo.mp4

# (Option D) Clean all generated scenes and index data for a complete reset
python tools/prepare_data.py --clean-all
```

### 4. Start the Services
First, start the FastAPI backend service:
```bash
# The environment variable is also required here
export KMP_DUPLICATE_LIB_OK=TRUE

uvicorn main:app --host 0.0.0.0 --port 8000
```

Then, in another terminal window, start the Gradio frontend UI:
```bash
# Activate the virtual environment and set the environment variable
source .venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE

# Start the UI
python ui.py
```

### 5. Start Using
Open your browser and go to the local URL provided by Gradio (usually `http://127.0.0.1:7860`). Your searches will now be performed across all the videos you have processed.

</details>
