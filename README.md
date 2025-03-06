# OVA
Voice_Assistant

# AI Chatbot with Real-time Voice using LLaMA & Streamlit

## 🚀 Overview
This project is an AI chatbot with real-time voice capabilities using:
- **LLaMA (llama.cpp)** for optimized inference
- **VITS (TTS API)** for real-time text-to-speech conversion
- **Streamlit** for an interactive web-based chat interface

## 📌 Features
✅ **Chat with AI:** Type your messages and get AI-generated responses.
✅ **Real-time Voice Output:** AI responses are converted into speech instantly.
✅ **CUDA-Optimized Processing:** Fully leverages NVIDIA GPUs for faster inference.
✅ **Maintains Chat History:** Keeps track of conversations dynamically.
✅ **User-Friendly UI:** Powered by Streamlit for a seamless chat experience.

## 🛠️ Installation
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/xJORDONx/OVA.git
cd OVA
```

### **2️⃣ Set Up Virtual Environment (Optional but Recommended)**
```sh
python -m venv venv  # Create a virtual environment
source venv/bin/activate  # Activate (Mac/Linux)
venv\Scripts\activate  # Activate (Windows)
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

## 🎯 Usage
### **Run the Streamlit Chatbot**
```sh
streamlit run app.py
```
This will start the chatbot in a web browser where you can interact with it.

## ⚙️ Configuration
Modify the `MODEL_PATH` in `app.py` to point to your LLaMA model file:
```python
MODEL_PATH = "Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_S.gguf"
```

## 🖥️ How It Works
1. User types a message in the chat input.
2. LLaMA generates a response based on the input.
3. The response is converted into speech using the VITS TTS model.
4. The speech is played in real-time using PyAudio.
5. The chat history is updated dynamically in Streamlit.

## 🏗️ Tech Stack
- **LLaMA (llama.cpp)** - Optimized AI inference
- **VITS TTS** - Fast text-to-speech processing
- **Streamlit** - Web UI for user interaction
- **PyAudio** - Real-time audio playback
- **Torch (CUDA-enabled)** - Accelerated AI computation

## 📌 Example Interaction
```
User: Hello AI!
AI: Hello! How can I assist you today?
(Audio plays AI's response)
```

## 🛠️ Troubleshooting
- **Issue: CUDA is not detected?**
  - Ensure you have installed PyTorch with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
  - Verify GPU support with `torch.cuda.is_available()`.
- **Issue: Model is too slow?**
  - Try reducing `max_tokens` or `n_ctx` in `app.py`.

## 📜 License
This project is open-source and licensed under the MIT License.

## 🌟 Contributing
Pull requests and issues are welcome! Feel free to enhance the chatbot's capabilities.

