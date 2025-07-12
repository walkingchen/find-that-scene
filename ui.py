import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/search"
MAX_RESULTS = 3

def search_scenes(query):
    # Prepare update objects for all possible components.
    # Start with them all being hidden.
    updates = [gr.update(visible=False)] * (MAX_RESULTS * 4) # 4 components per result group

    if not query:
        return tuple(updates)

    try:
        response = requests.get(API_URL, params={"query": query})
        response.raise_for_status()
        results = response.json()
        
        # For each result returned by the API, create a visible update
        for i in range(min(len(results), MAX_RESULTS)):
            r = results[i]
            base_idx = i * 4
            
            video_with_fragment = f"{r['video_path']}#t={r['start_time']},{r['end_time']}"
            info = (
                f"**视频源文件:** `{r['source_video_filename']}` "
                f"**文件:** `{r['keyframe_filename']}` "
                f"**场景:** {r['scene_id']} "
                f"**时间:** {r['start_time']:.2f}s - {r['end_time']:.2f}s\n"
            )

            # Update visibility and content for the group and its components
            updates[base_idx] = gr.update(visible=True)                 # Group
            updates[base_idx + 1] = gr.update(value=r['keyframe_path']) # Image
            updates[base_idx + 2] = gr.update(value=info)               # Markdown
            updates[base_idx + 3] = gr.update(value=video_with_fragment)# Video

        return tuple(updates)

    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return tuple(updates) # Return hidden updates on error


with gr.Blocks() as demo:
    gr.Markdown("# 文本驱动的视频镜头检索")
    gr.Markdown("输入一句话来描述你想要查找的视频场景，系统将返回最相关的 3 个镜头片段。")
    
    with gr.Row():
        text_input = gr.Textbox(label="输入描述文本", placeholder="例如：一个男人在黑暗中奔跑...", scale=3)
        search_button = gr.Button("搜索", scale=1)
    
    gr.Markdown("---")
    gr.Markdown("## 检索结果")

    result_components = []
    for i in range(MAX_RESULTS):
        with gr.Group(visible=False) as result_group:
            with gr.Row():
                with gr.Column():
                    keyframe_img = gr.Image(label=f"关键帧 {i+1}", interactive=False)
                    info_md = gr.Markdown(label=f"场景信息 {i+1}")
                with gr.Column():
                    video_player = gr.Video(label=f"播放片段 {i+1}", interactive=False)
        result_components.extend([result_group, keyframe_img, info_md, video_player])

    search_button.click(
        fn=search_scenes,
        inputs=text_input,
        outputs=result_components,
    )

if __name__ == "__main__":
    demo.launch(allowed_paths=["scenes", "video"])
