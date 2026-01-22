import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def handle_keyrelease(self, event):
    if self.editing_hist and self.multiline:
        self.text.tag_add('history', 'output_end', Tk_.INSERT)