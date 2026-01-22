import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import zstandard as zstd
from PIL import Image
from pathlib import Path
def select_file(self):
    file_path = filedialog.askopenfilename()
    if file_path:
        self.file_path_var.set(file_path)