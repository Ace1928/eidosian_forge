import tkinter as tk
from tkinter import filedialog, messagebox
import json
import logging
from typing import Callable, Dict, List, Optional, Tuple, Any
def create_toolbar(self) -> None:
    """Create a toolbar for quick access to common actions with iconographic buttons."""
    toolbar = tk.Frame(self.master, bd=1, relief=tk.RAISED)
    toolbar.pack(side=tk.TOP, fill=tk.X)
    buttons = [('New', self.new_project, '🆕'), ('Open', self.open_project, '📂'), ('Save', self.save_project, '💾'), ('Undo', self.undo, '↩️'), ('Redo', self.redo, '↪️'), ('Zoom In', self.zoom_in, '🔍++'), ('Zoom Out', self.zoom_out, '🔎--')]
    for text, command, icon in buttons:
        tk.Button(toolbar, text=f'{icon} {text}', command=command).pack(side=tk.LEFT, padx=2, pady=2)