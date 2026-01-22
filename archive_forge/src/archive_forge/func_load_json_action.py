import tkinter as tk
from tkinter import filedialog, scrolledtext
import requests
from bs4 import BeautifulSoup
import json
def load_json_action(self):
    file_path = filedialog.askopenfilename(filetypes=[('JSON files', '*.json')])
    if file_path:
        try:
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                self.text_area.insert(tk.END, json.dumps(json_data, indent=4))
        except Exception as e:
            self.text_area.insert(tk.END, f'Error loading JSON: {e}')