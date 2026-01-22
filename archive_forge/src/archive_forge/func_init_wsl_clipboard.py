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
def init_wsl_clipboard():

    def copy_wsl(text):
        text = _stringifyText(text)
        with subprocess.Popen(['clip.exe'], stdin=subprocess.PIPE, close_fds=True) as p:
            p.communicate(input=text.encode(ENCODING))

    def paste_wsl():
        with subprocess.Popen(['powershell.exe', '-command', 'Get-Clipboard'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True) as p:
            stdout = p.communicate()[0]
        return stdout[:-2].decode(ENCODING)
    return (copy_wsl, paste_wsl)