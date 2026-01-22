from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def is_wow64(self):
    """
        Determines if the thread is running under WOW64.

        @rtype:  bool
        @return:
            C{True} if the thread is running under WOW64. That is, it belongs
            to a 32-bit application running in a 64-bit Windows.

            C{False} if the thread belongs to either a 32-bit application
            running in a 32-bit Windows, or a 64-bit application running in a
            64-bit Windows.

        @raise WindowsError: On error an exception is raised.

        @see: U{http://msdn.microsoft.com/en-us/library/aa384249(VS.85).aspx}
        """
    try:
        wow64 = self.__wow64
    except AttributeError:
        if win32.bits == 32 and (not win32.wow64):
            wow64 = False
        else:
            wow64 = self.get_process().is_wow64()
        self.__wow64 = wow64
    return wow64