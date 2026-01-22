import sys
from winappdbg import win32
from winappdbg.system import System
from winappdbg.process import Process
from winappdbg.thread import Thread
from winappdbg.module import Module
from winappdbg.window import Window
from winappdbg.breakpoint import _BreakpointContainer, CodeBreakpoint
from winappdbg.event import Event, EventHandler, EventDispatcher, EventFactory
from winappdbg.interactive import ConsoleDebugger
import warnings
def is_debugee_started(self, dwProcessId):
    """
        Determine if the given process was started by the debugger.

        @see: L{is_debugee}, L{is_debugee_attached}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @rtype:  bool
        @return: C{True} if the given process was started for debugging by this
            L{Debug} instance.
        """
    return dwProcessId in self.__startedDebugees