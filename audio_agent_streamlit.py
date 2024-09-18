import threading
import httpx
import loguru
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



if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "Qwen/Qwen2-72B-Instruct"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    prompt_seach = LLMApi.build_prompt(prompt,search=True)
    st.session_state.messages.append(prompt_seach[0])
    st.session_state.messages.append(prompt_seach[1])
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = LLMApi.llm_client(llm_type="siliconflow").chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
        loguru.logger.info(f"response:{response}")
        execute_tts(response)
    st.session_state.messages.append({"role": "assistant", "content": response})