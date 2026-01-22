import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def report_callback_exception(self, exc, value, traceback):
    """
        Called when exceptions are caught by Tk.
        """
    sys.last_type = exc
    sys.last_value = value
    sys.last_traceback = traceback
    self.IP.showtraceback()