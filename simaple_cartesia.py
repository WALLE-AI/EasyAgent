from cartesia import Cartesia
import loguru
import pyaudio
import os

from dotenv import load_dotenv

from llm import LLMApi
load_dotenv()
client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))
voice_name = "starvlm"
voice_id = "a0e99841-438c-4a64-b679-ae501e7d6091"
voice = client.voices.get(id=voice_id)

def llm_transcript():
    test_text = "2024年9月，梵蒂冈城邦（Vatican City）元首，罗马天主教教皇方济各（Pope Francis）出访东南亚四国。9月13日，在从新加坡返回罗马的飞机上，教皇向随行记者们再次表达访问中国的意愿，并称对教廷与北京就续签有关主教任命临时协议的对话进展感到满意。"
    prompt = LLMApi.build_prompt(test_text,True)
    result = LLMApi.call_llm(prompt,llm_type="siliconflow",model_name="Qwen/Qwen2-7B-Instruct")
    loguru.logger.info(f"response result:{result.choices[0].message.content}")
    return result.choices[0].message.content

# transcript = llm_transcript()



def execute_tts(content):
    # You can check out our models at https://docs.cartesia.ai/getting-started/available-models
    model_id = "sonic-multilingual"

    # You can find the supported `output_format`s at https://docs.cartesia.ai/api-reference/endpoints/stream-speech-server-sent-events
    output_format = {
        "container": "raw",
        "encoding": "pcm_f32le",
        "sample_rate": 44100,
    }

    p = pyaudio.PyAudio()
    rate = 44100

    stream = None

    # Generate and stream audio
    for output in client.tts.sse(
        model_id=model_id,
        transcript=content,
        voice_embedding=voice["embedding"],
        stream=True,
        output_format=output_format,
    ):
        buffer = output["audio"]

        if not stream:
            stream = p.open(format=pyaudio.paFloat32, channels=1, rate=rate, output=True)

        # Write the audio data to the stream
        stream.write(buffer)

    stream.stop_stream()
    stream.close()
    p.terminate()