from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def disable_process_breakpoints(self, dwProcessId):
    """
        Disables all breakpoints for the given process.

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.
        """
    for bp in self.get_process_code_breakpoints(dwProcessId):
        self.disable_code_breakpoint(dwProcessId, bp.get_address())
    for bp in self.get_process_page_breakpoints(dwProcessId):
        self.disable_page_breakpoint(dwProcessId, bp.get_address())
    if self.system.has_process(dwProcessId):
        aProcess = self.system.get_process(dwProcessId)
    else:
        aProcess = Process(dwProcessId)
        aProcess.scan_threads()
    for aThread in aProcess.iter_threads():
        dwThreadId = aThread.get_tid()
        for bp in self.get_thread_hardware_breakpoints(dwThreadId):
            self.disable_hardware_breakpoint(dwThreadId, bp.get_address())