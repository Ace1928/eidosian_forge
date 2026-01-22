import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@staticmethod
def printable(data):
    """
        Replace unprintable characters with dots.

        @type  data: str
        @param data: Binary data.

        @rtype:  str
        @return: Printable text.
        """
    result = ''
    for c in data:
        if 32 < ord(c) < 128:
            result += c
        else:
            result += '.'
    return result