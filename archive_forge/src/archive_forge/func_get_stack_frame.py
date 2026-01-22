from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_stack_frame(self, max_size=None):
    """
        Reads the contents of the current stack frame.
        Only works for functions with standard prologue and epilogue.

        @type  max_size: int
        @param max_size: (Optional) Maximum amount of bytes to read.

        @rtype:  str
        @return: Stack frame data.
            May not be accurate, depending on the compiler used.
            May return an empty string.

        @raise RuntimeError: The stack frame is invalid,
            or the function doesn't have a standard prologue
            and epilogue.

        @raise WindowsError: An error occured when getting the thread context
            or reading data from the process memory.
        """
    sp, fp = self.get_stack_frame_range()
    size = fp - sp
    if max_size and size > max_size:
        size = max_size
    return self.get_process().peek(sp, size)