from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def read_stack_data(self, size=128, offset=0):
    """
        Reads the contents of the top of the stack.

        @type  size: int
        @param size: Number of bytes to read.

        @type  offset: int
        @param offset: Offset from the stack pointer to begin reading.

        @rtype:  str
        @return: Stack data.

        @raise WindowsError: Could not read the requested data.
        """
    aProcess = self.get_process()
    return aProcess.read(self.get_sp() + offset, size)