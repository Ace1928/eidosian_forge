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
def do_db(self, arg):
    """
        [~thread] db <register> - show memory contents as bytes
        [~thread] db <register-register> - show memory contents as bytes
        [~thread] db <register> <size> - show memory contents as bytes
        [~process] db <address> - show memory contents as bytes
        [~process] db <address-address> - show memory contents as bytes
        [~process] db <address> <size> - show memory contents as bytes
        """
    self.print_memory_display(arg, HexDump.hexblock)
    self.last_display_command = self.do_db