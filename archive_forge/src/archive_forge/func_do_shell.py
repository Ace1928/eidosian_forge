from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.system import System
from winappdbg.util import PathOperations
from winappdbg.event import EventHandler, NoEvent
from winappdbg.textio import HexInput, HexOutput, HexDump, CrashDump, DebugLog
import os
import sys
import code
import time
import warnings
import traceback
from cmd import Cmd
def do_shell(self, arg):
    """
        ! - spawn a system shell
        shell - spawn a system shell
        ! <command> [arguments...] - execute a single shell command
        shell <command> [arguments...] - execute a single shell command
        """
    if self.cmdprefix:
        raise CmdError('prefix not allowed')
    shell = os.getenv('ComSpec', 'cmd.exe')
    if arg:
        arg = '%s /c %s' % (shell, arg)
    else:
        arg = shell
    process = self.debug.system.start_process(arg, bConsole=True)
    process.wait()