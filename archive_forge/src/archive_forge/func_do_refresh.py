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
def do_refresh(self, arg):
    """
        refresh - refresh the list of running processes and threads
        [~process] refresh - refresh the list of running threads
        """
    if arg:
        raise CmdError('too many arguments')
    if self.cmdprefix:
        process = self.get_process_from_prefix()
        process.scan()
    else:
        self.debug.system.scan()