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
def decompress_and_show(self) -> None:
    """
        Initiates the process to decompress and display the stored image.
        """
    self.progress_bar.start()
    future = asyncio.run_coroutine_threadsafe(self.async_decompress_and_show(), self.asyncio_loop)
    future.add_done_callback(lambda x: self.after(0, self.on_decompress_and_show_done, x))