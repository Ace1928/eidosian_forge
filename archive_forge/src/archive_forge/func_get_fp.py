from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_fp(self):
    """
            @rtype:  int
            @return: Value of the frame pointer register.
            """
    flags = win32.CONTEXT_CONTROL | win32.CONTEXT_INTEGER
    context = self.get_context(flags)
    return context.fp