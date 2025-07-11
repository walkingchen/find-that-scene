import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/search"
BASE_URL = "http://127.0.0.1:8000"

def search_scenes(query):
    if not query:
        return []

    try:
        response = requests.get(API_URL, params={"query": query})
        response.raise_for_status()
        results = response.json()
        
        # Construct full URLs for the videos
        for r in results:
            r['video_url'] = f"{BASE_URL}{r['video_url']}"
            
        return [r['video_url'] for r in results]

    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return []

with gr.Blocks() as demo:
    gr.Markdown("# 文本驱动的视频镜头检索")
    gr.Markdown("输入一句话来描述你想要查找的视频场景，系统将返回最相关的 3 个镜头片段。")
    
    with gr.Row():
        text_input = gr.Textbox(label="输入描述文本", placeholder="例如：一个男人在黑暗中奔跑...")
        search_button = gr.Button("搜索")

    gallery = gr.Gallery(label="检索结果", elem_id="gallery", columns=3, height="auto")

    search_button.click(
        fn=search_scenes,
        inputs=text_input,
        outputs=gallery
    )

if __name__ == "__main__":
    demo.launch()
