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
def copy_qt(text):
    text = _stringifyText(text)
    cb = app.clipboard()
    cb.setText(text)