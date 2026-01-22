from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_all_hardware_breakpoints(self):
    """
        @rtype:  list of tuple( int, L{HardwareBreakpoint} )
        @return: All hardware breakpoints as a list of tuples (tid, bp).
        """
    result = list()
    for tid, bplist in compat.iteritems(self.__hardwareBP):
        for bp in bplist:
            result.append((tid, bp))
    return result