import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
def getWidth(self):
    """
        Get the width of the text output for the table.

        @rtype:  int
        @return: Width in characters for the text output,
            including the newline character.
        """
    width = 0
    if self.__width:
        width = sum((abs(x) for x in self.__width))
        width = width + len(self.__width) * len(self.__sep) + 1
    return width