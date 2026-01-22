from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_all_page_breakpoints(self):
    """
        @rtype:  list of tuple( int, L{PageBreakpoint} )
        @return: All page breakpoints as a list of tuples (pid, bp).
        """
    result = set()
    for (pid, address), bp in compat.iteritems(self.__pageBP):
        result.add((pid, bp))
    return list(result)