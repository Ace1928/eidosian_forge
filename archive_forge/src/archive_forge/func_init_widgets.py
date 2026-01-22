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
def init_widgets(self) -> None:
    self.load_btn = tk.Button(self, text='Load Image', command=self.load_image)
    self.load_btn.pack()
    self.compress_btn = tk.Button(self, text='Compress & Store Image', state='disabled', command=self.compress_and_store_image)
    self.compress_btn.pack()
    self.decompress_btn = tk.Button(self, text='Decompress & Show Image', command=self.decompress_and_show)
    self.decompress_btn.pack()
    self.browse_db_btn = tk.Button(self, text='Browse Stored Images', command=self.browse_stored_images)
    self.browse_db_btn.pack()
    self.original_img_label = tk.Label(self)
    self.original_img_label.pack(side='left', padx=10)
    self.decompressed_img_label = tk.Label(self)
    self.decompressed_img_label.pack(side='right', padx=10)
    self.status_label = tk.Label(self, text='')
    self.status_label.pack()
    self.progress_bar = ttk.Progressbar(self, orient='horizontal', mode='indeterminate')
    self.progress_bar.pack()
    self.pagination_frame = tk.Frame(self)
    self.prev_button = tk.Button(self.pagination_frame, text='Previous', command=self.previous_page)
    self.next_button = tk.Button(self.pagination_frame, text='Next', command=self.next_page)
    self.prev_button.pack(side=tk.LEFT, padx=10)
    self.next_button.pack(side=tk.RIGHT, padx=10)
    self.pagination_frame.pack(side=tk.BOTTOM, pady=10)
    self.image_display_frame = tk.Frame(self)
    self.image_display_frame.pack(fill=tk.BOTH, expand=True)
    self.image_labels = [tk.Label(self.image_display_frame) for _ in range(self.items_per_page)]
    for label in self.image_labels:
        label.pack()