import base64
from io import BytesIO, StringIO
import threading
import httpx
import loguru
from PIL import Image
from openai import OpenAI
import openai
from llm import LLMApi
import streamlit as st
import os
##加载env file
from dotenv import load_dotenv

from simaple_cartesia import execute_tts
load_dotenv()

st.title(":sunglasses: audioAgent")


if "messages" not in st.session_state:
        st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def image_to_base64(image_upload):
    mime_type = image_upload.type
    with Image.open(image_upload) as img:
        # 定义新的尺寸，例如缩小到原来的一半
        new_width = img.width // 2
        new_height = img.height // 2
        # 调整图片大小
        img_resized = img.resize((new_width, new_height))
        # 将图片转换为字节流
        buffered = BytesIO()
        img_resized.save(buffered, format=img.format)
        # 将字节流转换为Base64编码
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f'data:{mime_type};base64,{img_base64}'
    


with st.sidebar:
    option = st.selectbox('models selection',("qwen/qwen-2-vl-72b-instruct",'Qwen/Qwen2-72B-Instruct', 'Qwen/Qwen2-7B-Instruct',"starvlm-checkpoint1000","starchat-checkpoint1000","intern_vl"))

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = option
            
    upload_image = st.file_uploader("upload images", accept_multiple_files=False, type = ['jpg', 'png'])
    
    if upload_image:
        image = Image.open(upload_image)
        image_base64 =image_to_base64(upload_image)
        if prompt := st.chat_input("请输出您的问题"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            st.image(image, width=500)
            print(st.session_state["openai_model"])
            with st.chat_message("assistant"):
                message = LLMApi.build_image_prompt(prompt,image_base64)
                stream = LLMApi.llm_client(llm_type="openrouter").chat.completions.create(
                model=st.session_state["openai_model"],
                messages=message,
                stream=True,
            )
            response = st.write_stream(stream)       
    else:
        if prompt := st.chat_input("请输出您的问题"):
            prompt_seach = LLMApi.build_prompt(prompt,search=True)
            st.session_state.messages.append(prompt_seach[0])
            st.session_state.messages.append(prompt_seach[1])
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                stream = LLMApi.llm_client(llm_type="openrouter").chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                    stream=True,
                )
                response = st.write_stream(stream)
                execute_tts(response)
                loguru.logger.info(f"response:{response}")
            st.session_state.messages.append({"role": "assistant", "content": response})

