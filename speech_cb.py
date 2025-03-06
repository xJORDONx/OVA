import streamlit as st
import torch
import numpy as np
import torchaudio
import pyaudio
from llama_cpp import Llama  # ✅ Optimized LLaMA Inference
from TTS.api import TTS  # ✅ VITS for Real-time Speech

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LLAMA_FORCE_GPU"] = "1"

# **CONFIGURATION**
MODEL_PATH = "C:/Users/vishu/.lmstudio/models/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_S.gguf"

# ✅ **Load Optimized LLaMA Model (Using More GPU)**
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=150,  # 🔥 Fully offload to GPU (RTX 4080 can handle it)
    n_ctx=4096,        # ✅ Increase context for better understanding
    n_batch=256,       # ✅ Adjust batch size to avoid VRAM overflow
    use_mmap=True,
    use_mlock=True,
    use_trt=True
)

# ✅ **Enable CUDA Optimizations**
torch.backends.cuda.matmul.allow_tf32 = True  # 🔥 Enables TF32 for FASTER GPU Compute
torch.backends.cudnn.benchmark = True  # ✅ Optimize CUDA Kernel Selection
torch.cuda.synchronize()  # 🛠 Ensure CUDA Kernels Are Efficient

# ✅ **Load VITS TTS Model for Faster Speech**
tts = TTS(model_name="tts_models/en/ljspeech/vits").to("cuda" if torch.cuda.is_available() else "cpu")

# ✅ **Streamlit UI**
st.set_page_config(page_title="Chatbot with Voice", layout="wide")
st.title("💬 AI Chatbot with Real-time Voice")

# ✅ **Maintain Chat History**
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ **Display Chat History**
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ✅ **Generate AI Response with LLaMA**
def generate_response(user_input):
    try:
        # **Format Prompt**
        prompt = f"""### Instruction:
        You are a friendly AI assistant. Respond in a warm, engaging way.

        ### User:
        {user_input}

        ### AI Assistant:"""

        # **Generate Response**
        response = llm(
            prompt, 
            max_tokens=512, 
            temperature=0.8, 
            top_p=0.9, 
            repeat_penalty=1.1
        )
        ai_response = response["choices"][0]["text"].strip() if "choices" in response else "🤖: Sorry, I couldn't generate a response."

        # ✅ **Convert Text to Speech (No Saving, Just Play)**
        audio_samples = tts.tts(text=ai_response)
        audio_samples = np.array(audio_samples, dtype=np.float32)  # 🔥 Fix: Convert List to NumPy Array

        return ai_response, audio_samples

    except Exception as e:
        return f"🤖: Error generating response: {str(e)}", None

# ✅ **Real-time Audio Playback Using PyAudio**
def play_audio_realtime(audio_samples, sample_rate=22050):
    """Streams audio directly using PyAudio (Real-time Playback)"""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True)
    stream.write(audio_samples.tobytes())  # ✅ Fix: Convert NumPy array to bytes
    stream.stop_stream()
    stream.close()
    p.terminate()

# ✅ **Chat Interface**
user_input = st.chat_input("Type your message...")

if user_input:
    # ✅ **Append User Message**
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ✅ **Generate AI Response & Speech**
    response, audio_samples = generate_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # ✅ **Display AI Response & Speak in Real-Time**
    with st.chat_message("assistant"):
        st.markdown(response)

        # ✅ **Play Audio in Real-Time**
        if audio_samples is not None:
            play_audio_realtime(audio_samples)

    # ✅ **Refresh Streamlit Chat**
    st.rerun()
