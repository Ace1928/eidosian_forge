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
def copy_xclip(text, primary=False):
    text = _stringifyText(text)
    selection = DEFAULT_SELECTION
    if primary:
        selection = PRIMARY_SELECTION
    with subprocess.Popen(['xclip', '-selection', selection], stdin=subprocess.PIPE, close_fds=True) as p:
        p.communicate(input=text.encode(ENCODING))