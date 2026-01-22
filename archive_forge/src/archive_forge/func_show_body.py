import os
import sys
from .gui import *
from .app_menus import ListedWindow
def show_body(self):
    self.body_frame.grid(row=1, column=0, padx=5, pady=5, sticky=Tk_.N + Tk_.S + Tk_.W + Tk_.E)
    self.body_frame.focus_set()