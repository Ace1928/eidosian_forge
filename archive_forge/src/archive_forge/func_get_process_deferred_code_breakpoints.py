from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_process_deferred_code_breakpoints(self, dwProcessId):
    """
        Returns a list of deferred code breakpoints.

        @type  dwProcessId: int
        @param dwProcessId: Process ID.

        @rtype:  tuple of (int, str, callable, bool)
        @return: Tuple containing the following elements:
             - Label pointing to the address where to set the breakpoint.
             - Action callback for the breakpoint.
             - C{True} of the breakpoint is one-shot, C{False} otherwise.
        """
    return [(label, action, oneshot) for label, (action, oneshot) in compat.iteritems(self.__deferredBP.get(dwProcessId, {}))]