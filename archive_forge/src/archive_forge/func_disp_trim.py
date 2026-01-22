import os
import re
import sys
from functools import partial, partialmethod, wraps
from inspect import signature
from unicodedata import east_asian_width
from warnings import warn
from weakref import proxy
def disp_trim(data, length):
    """
    Trim a string which may contain ANSI control characters.
    """
    if len(data) == disp_len(data):
        return data[:length]
    ansi_present = bool(RE_ANSI.search(data))
    while disp_len(data) > length:
        data = data[:-1]
    if ansi_present and bool(RE_ANSI.search(data)):
        return data if data.endswith('\x1b[0m') else data + '\x1b[0m'
    return data