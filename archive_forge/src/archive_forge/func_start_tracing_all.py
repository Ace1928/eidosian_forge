from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def start_tracing_all(self):
    """
        Start tracing mode for all threads in all debugees.
        """
    for pid in self.get_debugee_pids():
        self.start_tracing_process(pid)