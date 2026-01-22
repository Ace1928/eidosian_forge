from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def clear_threads(self):
    """
        Clears the threads snapshot.
        """
    for aThread in compat.itervalues(self.__threadDict):
        aThread.clear()
    self.__threadDict = dict()