import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def edit_cut(self):
    try:
        self.text.clipboard_clear()
        self.text.clipboard_append(self.text.selection_get())
        self.text.delete(Tk_.SEL_FIRST, Tk_.SEL_LAST)
    except:
        pass