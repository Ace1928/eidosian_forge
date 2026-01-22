from winappdbg import win32
from winappdbg import compat
from winappdbg.system import System
from winappdbg.textio import HexDump, CrashDump
from winappdbg.util import StaticClass, MemoryAddresses, PathOperations
import sys
import os
import time
import zlib
import warnings
def getNotes(self):
    """
        Get the list of notes of this crash event.

        @rtype:  list( str )
        @return: List of notes.
        """
    return self.notes