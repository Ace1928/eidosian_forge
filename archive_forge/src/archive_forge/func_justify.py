import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
def justify(self, column, direction):
    """
        Make the text in a column left or right justified.

        @type  column: int
        @param column: Index of the column.

        @type  direction: int
        @param direction:
            C{-1} to justify left,
            C{1} to justify right.

        @raise IndexError: Bad column index.
        @raise ValueError: Bad direction value.
        """
    if direction == -1:
        self.__width[column] = abs(self.__width[column])
    elif direction == 1:
        self.__width[column] = -abs(self.__width[column])
    else:
        raise ValueError('Bad direction value.')