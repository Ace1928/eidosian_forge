from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging


model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(
    "/home/lloyd/Downloads/local_model_store/meta-llama/Meta-Llama-3-8B"
)
model.save_pretrained(
    "/home/lloyd/Downloads/local_model_store/meta-llama/Meta-Llama-3-8B"
)


## Check if the model and tokenizer are already downloaded
if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)
    logging.info("Model and tokenizer downloaded and saved.")
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    logging.info("Model and tokenizer loaded from local storage.")

# Set device based on your hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logging.info(f"Model moved to device: {device}")

# Generate text
prompt = "Once upon a time, in a far-off land"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
logging.info("Input encoded and moved to device.")

try:
    outputs = model.generate(
        inputs,
        max_new_tokens=500,
        top_p=0.95,
        do_sample=True,
        top_k=60,
        temperature=0.95,
        early_stopping=True,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info("Text generation successful.")
except Exception as e:
    logging.error(f"Error during text generation: {e}")
    generated_text = "Error in text generation."

print("Generated Text:", generated_text)
