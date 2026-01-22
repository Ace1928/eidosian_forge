from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_thread_ids(self):
    """
        @rtype:  list( int )
        @return: List of global thread IDs in this snapshot.
        """
    self.__initialize_snapshot()
    return compat.keys(self.__threadDict)