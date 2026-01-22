from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_traced_tids(self):
    """
        Retrieves the list of global IDs of all threads being traced.

        @rtype:  list( int... )
        @return: List of thread global IDs.
        """
    tids = list(self.__tracing)
    tids.sort()
    return tids