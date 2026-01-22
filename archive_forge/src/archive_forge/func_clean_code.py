import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def clean_code(self, code):
    """
        Remove blank lines and continuation prompts.
        """
    if not code.strip():
        return '\n'
    lines = [line for line in code.split('\n')]
    clean_lines = [lines[0].lstrip()]
    for line in lines[1:]:
        try:
            clean_lines.append(line.split(': ', 1)[1])
        except IndexError:
            pass
    return '\n'.join(clean_lines)