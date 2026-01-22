from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_stack_range(self):
    """
        @rtype:  tuple( int, int )
        @return: Stack beginning and end pointers, in memory addresses order.
            That is, the first pointer is the stack top, and the second pointer
            is the stack bottom, since the stack grows towards lower memory
            addresses.
        @raise   WindowsError: Raises an exception on error.
        """
    teb = self.get_teb()
    tib = teb.NtTib
    return (tib.StackLimit, tib.StackBase)