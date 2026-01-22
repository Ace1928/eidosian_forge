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
def nti(s):
    """Convert a number field to a python number.
    """
    if s[0] in (128, 255):
        n = 0
        for i in range(len(s) - 1):
            n <<= 8
            n += s[i + 1]
        if s[0] == 255:
            n = -(256 ** (len(s) - 1) - n)
    else:
        try:
            s = nts(s, 'ascii', 'strict')
            n = int(s.strip() or '0', 8)
        except ValueError:
            raise InvalidHeaderError('invalid header')
    return n