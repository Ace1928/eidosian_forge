from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def break_at(self, pid, address, action=None):
    """
        Sets a code breakpoint at the given process and address.

        If instead of an address you pass a label, the breakpoint may be
        deferred until the DLL it points to is loaded.

        @see: L{stalk_at}, L{dont_break_at}

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int or str
        @param address:
            Memory address of code instruction to break at. It can be an
            integer value for the actual address or a string with a label
            to be resolved.

        @type  action: function
        @param action: (Optional) Action callback function.

            See L{define_code_breakpoint} for more details.

        @rtype:  bool
        @return: C{True} if the breakpoint was set immediately, or C{False} if
            it was deferred.
        """
    bp = self.__set_break(pid, address, action, oneshot=False)
    return bp is not None