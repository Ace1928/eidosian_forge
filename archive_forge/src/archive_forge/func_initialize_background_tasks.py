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
def initialize_background_tasks(self) -> None:
    """
        Initializes background tasks necessary for the application, such as database initialization.
        """
    Thread(target=lambda: asyncio.run(self.async_init_db()), daemon=True).start()