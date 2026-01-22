import tkinter as tk
from tkinter import filedialog, messagebox
import json
import logging
from typing import Callable, Dict, List, Optional, Tuple, Any
def canvas_click(self, event: tk.Event) -> None:
    """Handle canvas click events to select or place widgets with detailed logging."""
    logging.debug(f'Canvas clicked at position: ({event.x}, {event.y})')