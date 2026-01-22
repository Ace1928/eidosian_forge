import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@staticmethod
def hexa_qword(data, separator=' '):
    """
        Convert binary data to a string of hexadecimal QWORDs.

        @type  data: str
        @param data: Binary data.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each QWORD.

        @rtype:  str
        @return: Hexadecimal representation.
        """
    if len(data) & 7 != 0:
        data += '\x00' * (8 - (len(data) & 7))
    return separator.join(['%.16x' % struct.unpack('<Q', data[i:i + 8])[0] for i in compat.xrange(0, len(data), 8)])