import sys
import os
import os.path
import ctypes
from ctypes import c_char_p, c_int, c_size_t, c_void_p, pointer, CFUNCTYPE, POINTER
import ctypes.util
import platform
import textwrap
def from_package_resources():
    if sys.platform != 'win32':
        return None
    bits, _ = platform.architecture()
    if bits == '64bit':
        subdir = 'mingw64'
    else:
        subdir = 'mingw32'
    this_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_path, 'data', subdir)
    find_message('looking in ', data_path)
    if os.path.exists(data_path):
        return from_prefix(data_path)