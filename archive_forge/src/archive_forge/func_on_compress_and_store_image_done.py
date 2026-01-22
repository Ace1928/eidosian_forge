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
def on_compress_and_store_image_done(self, future):
    """
        Callback function to be called when async_compress_and_store_image coroutine is done.
        """
    try:
        future.result()
        self.update_status('Image compressed, encrypted, and stored successfully.')
    except Exception as e:
        LoggingManager.error(f'Failed to compress, encrypt, and store image: {e}')
        self.update_status('Failed to compress, encrypt, and store image. Please try again.', error=True)
    finally:
        self.progress_bar.stop()
        self.compress_btn.config(state='normal')