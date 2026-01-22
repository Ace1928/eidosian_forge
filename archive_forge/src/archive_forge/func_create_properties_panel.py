import tkinter as tk
from tkinter import filedialog, messagebox
import json
import logging
from typing import Callable, Dict, List, Optional, Tuple, Any
def create_properties_panel(self) -> None:
    """Create a properties panel for editing widget properties with dynamic entry generation."""
    properties_panel = tk.Frame(self.master, bd=1, relief=tk.RAISED)
    properties_panel.pack(side=tk.RIGHT, fill=tk.Y)
    tk.Label(properties_panel, text='Properties').pack(pady=10)
    self.properties_entries: Dict[str, tk.Entry] = {}
    for prop in ['text', 'width', 'height', 'fg', 'bg', 'font', 'command', 'value', 'variable']:
        self.add_property_entry(properties_panel, prop)
    tk.Button(properties_panel, text='Apply', command=self.apply_properties).pack(pady=10)