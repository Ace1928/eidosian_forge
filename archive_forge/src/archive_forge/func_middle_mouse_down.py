import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def middle_mouse_down(self, event):
    if self.text.compare(Tk_.CURRENT, '<', 'output_end'):
        self.window.bell()
        try:
            self.nasty = str(self.text.index(Tk_.CURRENT))
        except AttributeError:
            return 'break'
        paste = event.widget.selection_get(selection='PRIMARY')
        self.nasty_text = paste.split()[0]
        return 'break'