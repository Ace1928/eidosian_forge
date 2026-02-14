import os
import logging
import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, render_template
from flask_cors import CORS

# Directory and model setup
model_name = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
model_dir = "/Aurora_M2/gradientllama3"

# Ensure the directory exists
os.makedirs(model_dir, exist_ok=True)

# Logging setup
logging.basicConfig(filename="gradient_chat_app.log", level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

# Load model and tokenizer
def load_model():
    try:
        if os.path.exists(model_dir):
            #model = AutoModelForCausalLM.from_pretrained(model_dir)
            #tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        logging.error("Error loading model: %s", e)
        raise

model, tokenizer = load_model()
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Flask app setup for Apache server
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    user_input = request.json.get("input")
    try:
        response = get_response(user_input)
        return {"response": response}
    except Exception as e:
        logging.error("Error generating response: %s", e)
        return {"error": "Failed to generate response"}, 500

def get_response(user_input):
    response = pipe(user_input, max_length=32000, num_return_sequences=1, truncation=True)[0]['generated_text']
    return response.strip()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# GUI setup
class ChatApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Chat with Llama-3-8B")
        self.create_widgets()
        self.context_length = 64000
        self.max_input_tokens = 8000
        self.max_new_tokens = 32000
        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 0.95

    def create_widgets(self):
        self.chat_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=20)
        self.chat_area.pack(padx=10, pady=10)
        self.chat_area.config(state=tk.DISABLED)

        self.entry = tk.Entry(self.root, width=80)
        self.entry.pack(padx=10, pady=10)
        self.entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.root, text="Send", command=self.send_message)
        self.send_button.pack(pady=5)

        self.config_button = tk.Button(self.root, text="Configure", command=self.configure_options)
        self.config_button.pack(pady=5)

    def send_message(self, event=None):
        user_input = self.entry.get()
        if user_input.strip():
            self.chat_area.config(state=tk.NORMAL)
            self.chat_area.insert(tk.END, f"User: {user_input}\n")
            self.chat_area.config(state=tk.DISABLED)

            response = self.get_response(user_input)
            self.chat_area.config(state=tk.NORMAL)
            self.chat_area.insert(tk.END, f"Llama-3-8B: {response}\n")
            self.chat_area.config(state=tk.DISABLED)

            self.entry.delete(0, tk.END)

    def get_response(self, user_input):
        try:
            response = pipe(user_input, max_length=self.max_new_tokens, num_return_sequences=1, truncation=True, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p)[0]['generated_text']
            return response.strip()
        except Exception as e:
            logging.error("Error getting response: %s", e)
            return "Sorry, I couldn't process your request."

    def configure_options(self):
        self.context_length = simpledialog.askinteger("Context Length", "Enter context length (default 64000):", initialvalue=self.context_length)
        self.max_input_tokens = simpledialog.askinteger("Max Input Tokens", "Enter max input tokens (default 8000):", initialvalue=self.max_input_tokens)
        self.max_new_tokens = simpledialog.askinteger("Max New Tokens", "Enter max new tokens (default 32000):", initialvalue=self.max_new_tokens)
        self.temperature = simpledialog.askfloat("Temperature", "Enter temperature (default 1.0):", initialvalue=self.temperature)
        self.top_k = simpledialog.askinteger("Top K", "Enter top_k (default 50):", initialvalue=self.top_k)
        self.top_p = simpledialog.askfloat("Top P", "Enter top_p (default 0.95):", initialvalue=self.top_p)

        messagebox.showinfo("Configuration", f"Context Length: {self.context_length}\nMax Input Tokens: {self.max_input_tokens}\nMax New Tokens: {self.max_new_tokens}\nTemperature: {self.temperature}\nTop K: {self.top_k}\nTop P: {self.top_p}")

# GUI main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApplication(root)
    root.mainloop()
