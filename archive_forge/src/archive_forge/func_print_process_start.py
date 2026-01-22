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
def print_process_start(self, event):
    pid = event.get_pid()
    start = event.get_start_address()
    if start:
        start = HexOutput.address(start)
        print('Started process %d at %s' % (pid, start))
    else:
        print('Attached to process %d' % pid)