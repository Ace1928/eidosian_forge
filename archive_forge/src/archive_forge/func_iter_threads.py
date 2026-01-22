from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def iter_threads(self):
    """
        @see:    L{iter_thread_ids}
        @rtype:  dictionary-valueiterator
        @return: Iterator of L{Thread} objects in this snapshot.
        """
    self.__initialize_snapshot()
    return compat.itervalues(self.__threadDict)