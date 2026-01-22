import tkinter as tk
from tkinter import filedialog, messagebox
import json
import logging
from typing import Callable, Dict, List, Optional, Tuple, Any
def save_project(self) -> None:
    """Save the current project to a file with detailed configuration saving."""
    logging.info('Saving the project.')
    file_path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON Files', '*.json')])
    if file_path:
        with open(file_path, 'w') as file:
            json.dump(self.config, file, indent=4)
            logging.debug(f'Project saved at: {file_path}')