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
将你的视频文件（例如 `my_video.mp4`）放入项目根目录下的 `video/` 文件夹中。

### 3. 数据预处理和索引
运行数据准备脚本，它会一次性完成视频切分、关键帧提取、特征生成和索引构建。请将 `video/demo.mp4` 替换为你的视频文件路径。
```bash
# 解决 OpenMP 库冲突 (主要针对 macOS)
export KMP_DUPLICATE_LIB_OK=TRUE

# 运行脚本
python tools/prepare_data.py --video_path video/demo.mp4
```
这个过程会需要一些时间，取决于视频大小和你的硬件性能。完成后，`scenes/` 目录会存放切分好的镜头片段和关键帧，`embeddings/` 目录会包含生成的向量索引。

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

python ui.py
```

### 5. 开始使用
打开浏览器，访问 Gradio 提供的本地 URL (通常是 `http://127.0.0.1:7860`)，即可开始使用。
