from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def set_condition(self, condition=True):
    """
        Sets a new condition callback for the breakpoint.

        @see: L{__init__}

        @type  condition: function
        @param condition: (Optional) Condition callback function.
        """
    if condition is None:
        self.__condition = True
    else:
        self.__condition = condition