from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def peek_stack_data(self, size=128, offset=0):
    """
        Tries to read the contents of the top of the stack.

        @type  size: int
        @param size: Number of bytes to read.

        @type  offset: int
        @param offset: Offset from the stack pointer to begin reading.

        @rtype:  str
        @return: Stack data.
            Returned data may be less than the requested size.
        """
    aProcess = self.get_process()
    return aProcess.peek(self.get_sp() + offset, size)