import contextlib
import ctypes
from ctypes import (
from ctypes.util import find_library
def cfstr_to_str(data):
    return ctypes.string_at(CFDataGetBytePtr(data), CFDataGetLength(data)).decode('utf-8')