from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def erase_page_breakpoint(self, dwProcessId, address):
    """
        Erases the page breakpoint at the given address.

        @see:
            L{define_page_breakpoint},
            L{has_page_breakpoint},
            L{get_page_breakpoint},
            L{enable_page_breakpoint},
            L{enable_one_shot_page_breakpoint},
            L{disable_page_breakpoint}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address of breakpoint.
        """
    bp = self.get_page_breakpoint(dwProcessId, address)
    begin = bp.get_address()
    end = begin + bp.get_size()
    if not bp.is_disabled():
        self.disable_page_breakpoint(dwProcessId, address)
    address = begin
    pageSize = MemoryAddresses.pageSize
    while address < end:
        del self.__pageBP[dwProcessId, address]
        address = address + pageSize