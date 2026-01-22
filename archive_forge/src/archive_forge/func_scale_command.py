import sys
import tkinter
from tkinter import ttk
from .zoom_slider import Slider
import time
def scale_command(self, value):
    self.set_value(value)
    self.update_label()
    if self.update_function:
        self.update_function()