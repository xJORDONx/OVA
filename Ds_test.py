import streamlit as st
import llama_cpp

# Load Model
model_path = "C:/Users/vishu/.lmstudio/models/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_S.gguf"
llm = llama_cpp.Llama(model_path=model_path, n_ctx=256, n_batch=512, n_gpu_layers=32,use_mmap= True,use_mlock=True)

# Streamlit UI
st.title("🤖 LLaMA-2 AI Assistant")
st.write("Chat with your local AI model.")

if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
user_input = st.text_input("You:", "")

if user_input:
    response = llm(user_input, max_tokens=200, temperature=0.7, top_p=0.9, echo=False)
    bot_response = response["choices"][0]["text"].strip()
    
    # Store conversation history
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("AI", bot_response))

# Display chat history
for role, text in st.session_state.history:
    st.write(f"**{role}:** {text}")
