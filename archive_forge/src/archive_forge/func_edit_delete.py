import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def edit_delete(self):
    try:
        self.text.delete(Tk_.SEL_FIRST, Tk_.SEL_LAST)
    except:
        pass