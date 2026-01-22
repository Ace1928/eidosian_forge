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
def print_breakpoint_location(self, event):
    process = event.get_process()
    thread = event.get_thread()
    pc = event.get_exception_address()
    self.print_current_location(process, thread, pc)