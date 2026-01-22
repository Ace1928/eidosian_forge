import asyncio
import io
import tkinter as tk
import threading
from threading import Thread
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from cryptography.fernet import Fernet
from core_services import LoggingManager, EncryptionManager
from database import (
from image_processing import (
def update_image_display(self, metadata_list):
    for label in self.image_labels:
        label.config(image='')
        label.image = None
    for metadata, label in zip(metadata_list, self.image_labels):
        try:
            img_data = metadata[0]
            img = Image.open(io.BytesIO(img_data))
            img.thumbnail((100, 100), Image.ANTIALIAS)
            tk_img = ImageTk.PhotoImage(img)
            label.config(image=tk_img)
            label.image = tk_img
        except Exception as e:
            LoggingManager.error(f'Error displaying image: {e}')