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
def do_trace(self, arg):
    """
        t - trace at the current assembly instruction
        trace - trace at the current assembly instruction
        """
    if arg:
        raise CmdError('too many arguments')
    if self.lastEvent is None:
        raise CmdError('no current thread set')
    self.lastEvent.get_thread().set_tf()
    return True