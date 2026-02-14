import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tkinter as tk
from tkinter import scrolledtext, messagebox
import time
import os

# Set up paths and log management
LOG_DIR = 'chat_logs'
os.makedirs(LOG_DIR, exist_ok=True)

def get_log_file():
    current_time = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(LOG_DIR, f'chat_log_{current_time}.txt')

log_file = get_log_file()

def check_log_size():
    global log_file
    if os.path.exists(log_file) and os.path.getsize(log_file) >= 10 * 1024 * 1024:
        log_file = get_log_file()

def log_conversation(role, message):
    check_log_size()
    with open(log_file, 'a') as f:
        f.write(f"{role}: {message}\n")

# Initialize model and tokenizer
model_name = "gpt2"  # You can change this to another model if desired
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Transformer Chatbot")
        self.create_widgets()

    def create_widgets(self):
        self.chat_display = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='disabled')
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.input_field = tk.Entry(self.root, width=100)
        self.input_field.grid(row=1, column=0, padx=10, pady=10)
        self.input_field.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.root, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        self.system_prompt = tk.Text(self.root, height=3, width=100, wrap=tk.WORD)
        self.system_prompt.insert(tk.END, "You are a helpful assistant.")
        self.system_prompt.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        self.compare_button = tk.Button(self.root, text="Compare CPU vs CUDA", command=self.compare_performance)
        self.compare_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    def send_message(self, event=None):
        user_input = self.input_field.get()
        if user_input.strip() == "":
            return
        self.display_message("User", user_input)
        self.input_field.delete(0, tk.END)
        log_conversation("User", user_input)

        response = self.generate_response(user_input)
        self.display_message("AI", response)
        log_conversation("AI", response)

    def display_message(self, sender, message):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, f"{sender}: {message}\n")
        self.chat_display.yview(tk.END)
        self.chat_display.config(state='disabled')

    def generate_response(self, user_input):
        system_prompt = self.system_prompt.get("1.0", tk.END).strip()
        prompt = f"{system_prompt}\nUser: {user_input}\nAI:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("AI:")[-1].strip()

    def compare_performance(self):
        user_input = "Tell me a joke."
        system_prompt = self.system_prompt.get("1.0", tk.END).strip()
        prompt = f"{system_prompt}\nUser: {user_input}\nAI:"
        inputs = tokenizer(prompt, return_tensors="pt")

        start_time = time.time()
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
        device_time = time.time() - start_time

        if torch.cuda.is_available():
            start_time = time.time()
            with torch.no_grad():
                inputs = {k: v.to('cpu') for k, v in inputs.items()}
                outputs = model.generate(**inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
            cpu_time = time.time() - start_time
        else:
            cpu_time = "N/A"

        messagebox.showinfo("Performance Comparison", f"Device Time (CUDA if available, else CPU): {device_time:.2f} seconds\nCPU Time: {cpu_time if cpu_time == 'N/A' else f'{cpu_time:.2f} seconds'}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
