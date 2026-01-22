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
def start_asyncio_loop(self):
    """Runs the asyncio event loop."""
    asyncio.set_event_loop(self.asyncio_loop)
    self.asyncio_loop.run_forever()