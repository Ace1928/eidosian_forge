from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_stack_frame_range(self):
    """
        Returns the starting and ending addresses of the stack frame.
        Only works for functions with standard prologue and epilogue.

        @rtype:  tuple( int, int )
        @return: Stack frame range.
            May not be accurate, depending on the compiler used.

        @raise RuntimeError: The stack frame is invalid,
            or the function doesn't have a standard prologue
            and epilogue.

        @raise WindowsError: An error occured when getting the thread context.
        """
    st, sb = self.get_stack_range()
    sp = self.get_sp()
    fp = self.get_fp()
    size = fp - sp
    if not st <= sp < sb:
        raise RuntimeError('Stack pointer lies outside the stack')
    if not st <= fp < sb:
        raise RuntimeError('Frame pointer lies outside the stack')
    if sp > fp:
        raise RuntimeError('No valid stack frame found')
    return (sp, fp)