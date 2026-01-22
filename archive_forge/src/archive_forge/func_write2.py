import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def write2(self, string):
    """
        Write method for messages.  These go in the "immutable"
        part, so as not to confuse the prompt.
        """
    self.window.tkraise()
    self.text.mark_set('save_insert', Tk_.INSERT)
    self.text.mark_set('save_end', 'output_end')
    self.text.mark_set(Tk_.INSERT, str(self.text.index('output_end')) + 'linestart')
    self.text.insert(Tk_.INSERT, string, ('output', 'msg'))
    self.text.mark_set(Tk_.INSERT, 'save_insert')
    self.text.see('output_end')