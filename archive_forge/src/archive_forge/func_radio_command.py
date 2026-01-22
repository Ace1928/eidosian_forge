import sys
import tkinter
from tkinter import ttk
from .zoom_slider import Slider
import time
def radio_command(self):
    self.set_value(self.radio_var.get())
    if self.update_function:
        self.update_function()