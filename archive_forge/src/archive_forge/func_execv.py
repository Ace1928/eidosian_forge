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
def execv(self, argv, **kwargs):
    """
        Starts a new process for debugging.

        This method uses a list of arguments. To use a command line string
        instead, use L{execl}.

        @see: L{attach}, L{detach}

        @type  argv: list( str... )
        @param argv: List of command line arguments to pass to the debugee.
            The first element must be the debugee executable filename.

        @type    bBreakOnEntryPoint: bool
        @keyword bBreakOnEntryPoint: C{True} to automatically set a breakpoint
            at the program entry point.

        @type    bConsole: bool
        @keyword bConsole: True to inherit the console of the debugger.
            Defaults to C{False}.

        @type    bFollow: bool
        @keyword bFollow: C{True} to automatically attach to child processes.
            Defaults to C{False}.

        @type    bInheritHandles: bool
        @keyword bInheritHandles: C{True} if the new process should inherit
            it's parent process' handles. Defaults to C{False}.

        @type    bSuspended: bool
        @keyword bSuspended: C{True} to suspend the main thread before any code
            is executed in the debugee. Defaults to C{False}.

        @keyword dwParentProcessId: C{None} or C{0} if the debugger process
            should be the parent process (default), or a process ID to
            forcefully set as the debugee's parent (only available for Windows
            Vista and above).

            In hostile mode, the default is not the debugger process but the
            process ID for "explorer.exe".

        @type    iTrustLevel: int or None
        @keyword iTrustLevel: Trust level.
            Must be one of the following values:
             - 0: B{No trust}. May not access certain resources, such as
                  cryptographic keys and credentials. Only available since
                  Windows XP and 2003, desktop editions. This is the default
                  in hostile mode.
             - 1: B{Normal trust}. Run with the same privileges as a normal
                  user, that is, one that doesn't have the I{Administrator} or
                  I{Power User} user rights. Only available since Windows XP
                  and 2003, desktop editions.
             - 2: B{Full trust}. Run with the exact same privileges as the
                  current user. This is the default in normal mode.

        @type    bAllowElevation: bool
        @keyword bAllowElevation: C{True} to allow the child process to keep
            UAC elevation, if the debugger itself is running elevated. C{False}
            to ensure the child process doesn't run with elevation. Defaults to
            C{True}.

            This flag is only meaningful on Windows Vista and above, and if the
            debugger itself is running with elevation. It can be used to make
            sure the child processes don't run elevated as well.

            This flag DOES NOT force an elevation prompt when the debugger is
            not running with elevation.

            Note that running the debugger with elevation (or the Python
            interpreter at all for that matter) is not normally required.
            You should only need to if the target program requires elevation
            to work properly (for example if you try to debug an installer).

        @rtype:  L{Process}
        @return: A new Process object. Normally you don't need to use it now,
            it's best to interact with the process from the event handler.

        @raise WindowsError: Raises an exception on error.
        """
    if type(argv) in (str, compat.unicode):
        raise TypeError('Debug.execv expects a list, not a string')
    lpCmdLine = self.system.argv_to_cmdline(argv)
    return self.execl(lpCmdLine, **kwargs)