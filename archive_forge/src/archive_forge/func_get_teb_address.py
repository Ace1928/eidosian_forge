from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_teb_address(self):
    """
        Returns a remote pointer to the TEB.

        @rtype:  int
        @return: Remote pointer to the L{TEB} structure.
        @raise WindowsError: An exception is raised on error.
        """
    try:
        return self._teb_ptr
    except AttributeError:
        try:
            hThread = self.get_handle(win32.THREAD_QUERY_INFORMATION)
            tbi = win32.NtQueryInformationThread(hThread, win32.ThreadBasicInformation)
            address = tbi.TebBaseAddress
        except WindowsError:
            address = self.get_linear_address('SegFs', 0)
            if not address:
                raise
        self._teb_ptr = address
        return address