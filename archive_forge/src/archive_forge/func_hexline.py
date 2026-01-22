import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@classmethod
def hexline(cls, data, separator=' ', width=None):
    """
        Dump a line of hexadecimal numbers from binary data.

        @type  data: str
        @param data: Binary data.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each character.

        @type  width: int
        @param width:
            (Optional) Maximum number of characters to convert per text line.
            This value is also used for padding.

        @rtype:  str
        @return: Multiline output text.
        """
    if width is None:
        fmt = '%s  %s'
    else:
        fmt = '%%-%ds  %%-%ds' % ((len(separator) + 2) * width - 1, width)
    return fmt % (cls.hexadecimal(data, separator), cls.printable(data))