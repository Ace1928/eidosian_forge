from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def eval_condition(self, event):
    """
        Evaluates the breakpoint condition, if any was set.

        @type  event: L{Event}
        @param event: Debug event triggered by the breakpoint.

        @rtype:  bool
        @return: C{True} to dispatch the event, C{False} otherwise.
        """
    condition = self.get_condition()
    if condition is True:
        return True
    if callable(condition):
        try:
            return bool(condition(event))
        except Exception:
            e = sys.exc_info()[1]
            msg = 'Breakpoint condition callback %r raised an exception: %s'
            msg = msg % (condition, traceback.format_exc(e))
            warnings.warn(msg, BreakpointCallbackWarning)
            return False
    return bool(condition)