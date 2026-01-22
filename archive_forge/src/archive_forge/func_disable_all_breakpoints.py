from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def disable_all_breakpoints(self):
    """
        Disables all breakpoints in all processes.

        @see:
            disable_code_breakpoint,
            disable_page_breakpoint,
            disable_hardware_breakpoint
        """
    for pid, bp in self.get_all_code_breakpoints():
        self.disable_code_breakpoint(pid, bp.get_address())
    for pid, bp in self.get_all_page_breakpoints():
        self.disable_page_breakpoint(pid, bp.get_address())
    for tid, bp in self.get_all_hardware_breakpoints():
        self.disable_hardware_breakpoint(tid, bp.get_address())