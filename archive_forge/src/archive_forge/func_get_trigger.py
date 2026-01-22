from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_trigger(self):
    """
        @see: L{validTriggers}
        @rtype:  int
        @return: The breakpoint trigger flag.
        """
    return self.__trigger