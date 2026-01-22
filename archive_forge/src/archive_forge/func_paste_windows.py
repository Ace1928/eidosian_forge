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
def paste_windows():
    with clipboard(None):
        handle = safeGetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ''
        return c_wchar_p(handle).value