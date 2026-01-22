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
def init_wl_clipboard():
    PRIMARY_SELECTION = '-p'

    def copy_wl(text, primary=False):
        text = _stringifyText(text)
        args = ['wl-copy']
        if primary:
            args.append(PRIMARY_SELECTION)
        if not text:
            args.append('--clear')
            subprocess.check_call(args, close_fds=True)
        else:
            p = subprocess.Popen(args, stdin=subprocess.PIPE, close_fds=True)
            p.communicate(input=text.encode(ENCODING))

    def paste_wl(primary=False):
        args = ['wl-paste', '-n']
        if primary:
            args.append(PRIMARY_SELECTION)
        p = subprocess.Popen(args, stdout=subprocess.PIPE, close_fds=True)
        stdout, _stderr = p.communicate()
        return stdout.decode(ENCODING)
    return (copy_wl, paste_wl)