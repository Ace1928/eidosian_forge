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
def do_gn(self, arg):
    """
        gn - go with exception not handled
        """
    if self.cmdprefix:
        raise CmdError('prefix not allowed')
    if arg:
        raise CmdError('too many arguments')
    if self.lastEvent:
        self.lastEvent.continueStatus = win32.DBG_EXCEPTION_NOT_HANDLED
    return self.do_go(arg)