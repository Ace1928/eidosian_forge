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
def browse_stored_images(self) -> None:
    """
        Initiates the browsing of stored images starting from the first page.
        """
    self.current_page = 0
    self.async_browse_stored_images()