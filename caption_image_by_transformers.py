import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor,BlipForConditionalGeneration

## pip install langchain==0.1.11 gradio==5.23.2 transformers==4.38.2 bs4==0.0.2 requests==2.31.0 torch==2.2.1

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def caption_image(input_image:np.ndarray):
    raw_image = Image.fromarray(input_image).convert("RGB")
    inputs = processor(raw_image,return_tensors="pt")
    out = model.generate(**inputs,max_length=50)
    caption = processor.decode(out[0],skip_special_tokens=True)
    return caption


iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs="text",
    title="图片标题生成",
    description="给定一张图片，生成对应的标题",
    theme="compact",
)

iface.launch()

