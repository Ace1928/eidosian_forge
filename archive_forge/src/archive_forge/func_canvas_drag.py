import tkinter as tk
from tkinter import filedialog, messagebox
import json
import logging
from typing import Callable, Dict, List, Optional, Tuple, Any
def canvas_drag(self, event: tk.Event) -> None:
    """Handle canvas drag events to move widgets or draw with detailed logging."""
    logging.debug(f'Canvas dragged to position: ({event.x}, {event.y})')