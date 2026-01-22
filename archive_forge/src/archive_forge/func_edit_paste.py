import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def edit_paste(self, event=None):
    text = self.text
    try:
        clip = self.text.clipboard_get()
    except:
        return
    try:
        start = text.index(Tk_.SEL_FIRST)
        text.delete(Tk_.SEL_FIRST, Tk_.SEL_LAST)
        text.insert(Tk_.SEL_FIRST, clip)
    except:
        if self.text.compare(Tk_.INSERT, '<', 'output_end'):
            self.text.mark_set(Tk_.INSERT, 'output_end')
        text.insert(Tk_.INSERT, clip)
        self.text.see(Tk_.INSERT)
    try:
        self.text.tag_remove(Tk_.SEL, Tk_.SEL_FIRST, Tk_.SEL_LAST)
    except:
        pass
    return 'break'