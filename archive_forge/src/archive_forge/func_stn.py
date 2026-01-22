from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def stn(s, length, encoding, errors):
    """Convert a string to a null-terminated bytes object.
    """
    if s is None:
        raise ValueError('metadata cannot contain None')
    s = s.encode(encoding, errors)
    return s[:length] + (length - len(s)) * NUL