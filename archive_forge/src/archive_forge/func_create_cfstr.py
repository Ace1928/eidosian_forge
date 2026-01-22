import contextlib
import ctypes
from ctypes import (
from ctypes.util import find_library
def create_cfstr(s):
    return CFStringCreateWithCString(None, s.encode('utf8'), 134217984)