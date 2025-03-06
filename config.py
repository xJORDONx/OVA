import torch
from llama_cpp import Llama

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load DeepSeek model
model_path = "C:/Users/vishu/.lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
llm = Llama(model_path=model_path, n_gpu_layers=-1)  # -1 for full GPU usage

# Test inference
response = llm("What is today's date?")
print("DeepSeek Response:", response["choices"][0]["text"])
