import tkinter as tk
from tkinter import filedialog, messagebox
import json
import logging
from typing import Callable, Dict, List, Optional, Tuple, Any
def open_project(self) -> None:
    """Open an existing project from a file with detailed configuration loading."""
    logging.info('Opening a project.')
    file_path = filedialog.askopenfilename(filetypes=[('JSON Files', '*.json')])
    if file_path:
        with open(file_path, 'r') as file:
            config = json.load(file)
            logging.debug(f'Project configuration loaded: {config}')