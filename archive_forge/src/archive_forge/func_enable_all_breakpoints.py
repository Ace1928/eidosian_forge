from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def enable_all_breakpoints(self):
    """
        Enables all disabled breakpoints in all processes.

        @see:
            enable_code_breakpoint,
            enable_page_breakpoint,
            enable_hardware_breakpoint
        """
    for pid, bp in self.get_all_code_breakpoints():
        if bp.is_disabled():
            self.enable_code_breakpoint(pid, bp.get_address())
    for pid, bp in self.get_all_page_breakpoints():
        if bp.is_disabled():
            self.enable_page_breakpoint(pid, bp.get_address())
    for tid, bp in self.get_all_hardware_breakpoints():
        if bp.is_disabled():
            self.enable_hardware_breakpoint(tid, bp.get_address())