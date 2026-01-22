import sys
import os
import os.path
import ctypes
from ctypes import c_char_p, c_int, c_size_t, c_void_p, pointer, CFUNCTYPE, POINTER
import ctypes.util
import platform
import textwrap
def cbfunc1(tag, name, desc, file, data):
    cbfunc(tag, name, desc, file)