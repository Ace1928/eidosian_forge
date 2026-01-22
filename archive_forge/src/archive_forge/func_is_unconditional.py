from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def is_unconditional(self):
    """
        @rtype:  bool
        @return: C{True} if the breakpoint doesn't have a condition callback defined.
        """
    return self.__condition is True