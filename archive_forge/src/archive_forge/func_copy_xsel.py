import contextlib
import ctypes
from ctypes import (
import os
import platform
from shutil import which as _executable_exists
import subprocess
import time
import warnings
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
def copy_xsel(text, primary=False):
    text = _stringifyText(text)
    selection_flag = DEFAULT_SELECTION
    if primary:
        selection_flag = PRIMARY_SELECTION
    with subprocess.Popen(['xsel', selection_flag, '-i'], stdin=subprocess.PIPE, close_fds=True) as p:
        p.communicate(input=text.encode(ENCODING))