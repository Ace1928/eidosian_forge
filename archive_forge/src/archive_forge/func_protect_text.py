import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def protect_text(self, event):
    try:
        if self.text.compare(Tk_.SEL_FIRST, '<=', 'output_end'):
            self.window.bell()
            return 'break'
    except:
        pass