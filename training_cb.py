from vllm import LLM, SamplingParams
import torch
import numpy as np
import pyaudio
from TTS.api import TTS  # ✅ VITS for Real-time Speech

# ✅ **Set up Model Configuration**
MODEL_NAME = "HuggingFaceH4/llama-chat-hf"  # 🔥 Change this to Llamac-chat-HF model

# ✅ **Use vLLM for Super-Fast Inference**
llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=1,  # Set according to your GPU setup
    gpu_memory_utilization=0.9,  # 🔥 Optimize GPU usage
    dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16"
)

# ✅ **Configure Sampling for Faster Responses**
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=128,  # ✅ Shorten response time
    repetition_penalty=1.1
)

# ✅ **Load VITS TTS Model**
tts = TTS(model_name="tts_models/en/ljspeech/vits").to("cuda" if torch.cuda.is_available() else "cpu")

# ✅ **Generate AI Response with vLLM**
def generate_response(user_input):
    try:
        prompt = f"### Instruction:\nYou are a helpful AI assistant.\n\n### User:\n{user_input}\n\n### AI Assistant:"

        # 🔥 **Use vLLM for Fast Inference**
        outputs = llm.generate([prompt], sampling_params)
        ai_response = outputs[0].outputs[0].text.strip()

        # ✅ **Convert Text to Speech**
        audio_samples = tts.tts(text=ai_response)
        audio_samples = np.array(audio_samples, dtype=np.float32).flatten()  # 🔥 Fix: Flatten tensor
        
        return ai_response, audio_samples

    except Exception as e:
        return f"🤖 Error: {str(e)}", None

# ✅ **Real-time Audio Playback**
def play_audio_realtime(audio_samples, sample_rate=22050):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True)
    stream.write(audio_samples.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

# ✅ **Simple Chat Loop**
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("🤖 Goodbye!")
            break
        
        response, audio_samples = generate_response(user_input)
        print(f"AI: {response}")

        # ✅ **Play AI Speech in Real-Time**
        if audio_samples is not None:
            play_audio_realtime(audio_samples)
