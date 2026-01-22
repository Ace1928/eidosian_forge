from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_process_code_breakpoints(self, dwProcessId):
    """
        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @rtype:  list of L{CodeBreakpoint}
        @return: All code breakpoints for the given process.
        """
    return [bp for (pid, address), bp in compat.iteritems(self.__codeBP) if pid == dwProcessId]