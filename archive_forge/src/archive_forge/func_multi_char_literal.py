import copy
import ctypes
import ctypes.util
import os
import sys
from .exceptions import DecodeError
from .base import AudioFile
def multi_char_literal(chars):
    """Emulates character integer literals in C. Given a string "abc",
    returns the value of the C single-quoted literal 'abc'.
    """
    num = 0
    for index, char in enumerate(chars):
        shift = (len(chars) - index - 1) * 8
        num |= ord(char) << shift
    return num