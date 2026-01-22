from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_size_in_pages(self):
    """
        @rtype:  int
        @return: The size in pages of the breakpoint.
        """
    return self.get_size() // MemoryAddresses.pageSize