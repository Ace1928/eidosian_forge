import os
import io
import re
import sys
import tempfile
import subprocess
from io import UnsupportedOperation
from pathlib import Path
from IPython import get_ipython
from IPython.display import display
from IPython.core.error import TryNext
from IPython.utils.data import chop
from IPython.utils.process import system
from IPython.utils.terminal import get_terminal_size
from IPython.utils import py3compat
def display_page(strng, start=0, screen_lines=25):
    """Just display, no paging. screen_lines is ignored."""
    if isinstance(strng, dict):
        data = strng
    else:
        if start:
            strng = u'\n'.join(strng.splitlines()[start:])
        data = {'text/plain': strng}
    display(data, raw=True)