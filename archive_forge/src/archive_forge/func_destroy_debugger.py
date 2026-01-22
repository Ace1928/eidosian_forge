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
def destroy_debugger(self, autodetach=True):
    debug = self.stop_using_debugger()
    if debug is not None:
        if not autodetach:
            debug.kill_all(bIgnoreExceptions=True)
            debug.lastEvent = None
        debug.stop()
    del debug