# pip install transformers==4.36.0 torch==2.1.1 gradio==5.23.2 langchain==0.0.343 ibm_watson_machine_learning==1.0.335 huggingface-hub==0.28.1

import torch
import os
import gradio as gr

#from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub

from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

my_credentials = {
    "url"    : "https://us-south.ml.cloud.ibm.com"
}
params = {
        GenParams.MAX_NEW_TOKENS: 800, # 模型在单次运行中可以生成的最大令牌数。
        GenParams.TEMPERATURE: 0.1,   # 控制令牌生成随机性的参数。较低的值使生成更具确定性，而较高的值则引入更多随机性。
    }

LLAMA2_model = Model(
        model_id= 'meta-llama/llama-3-2-11b-vision-instruct', 
        credentials=my_credentials,
        params=params,
        project_id="skills-network",  
        )

llm = WatsonxLLM(LLAMA2_model)  

#######------------- 提示模板-------------####

temp = """
<s><<SYS>>
列出上下文中的关键点和详细信息： 
[INST] 上下文 : {context} [/INST] 
<</SYS>>
"""

pt = PromptTemplate(
    input_variables=["context"],
    template= temp)

prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)

#######------------- 语音转文本-------------####

def transcript_audio(audio_file):
    # 初始化语音识别管道
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    # 转录音频文件并返回结果
    transcript_txt = pipe(audio_file, batch_size=8)["text"]
    result = prompt_to_LLAMA2.run(transcript_txt)

    return result

#######------------- Gradio-------------####

audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

iface = gr.Interface(fn= transcript_audio, 
                    inputs= audio_input, outputs= output_text, 
                    title= "音频转录应用",
                    description= "上传音频文件")

iface.launch(server_name="0.0.0.0", server_port=7860)
