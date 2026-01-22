import tkinter as tk
from tkinter import filedialog, messagebox
import json
import logging
from typing import Callable, Dict, List, Optional, Tuple, Any
def show_about(self) -> None:
    """Display an 'About' dialog with detailed version and developer information."""
    messagebox.showinfo('About', 'Universal GUI Builder\nVersion 1.0\nDeveloped by Neuro Forge')