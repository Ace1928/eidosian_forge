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
def do_stack(self, arg):
    """
        [~thread] k - show the stack trace
        [~thread] stack - show the stack trace
        """
    if arg:
        raise CmdError('too many arguments')
    pid, tid = self.get_process_and_thread_ids_from_prefix()
    process = self.get_process(pid)
    thread = process.get_thread(tid)
    try:
        stack_trace = thread.get_stack_trace_with_labels()
        if stack_trace:
            print(CrashDump.dump_stack_trace_with_labels(stack_trace))
        else:
            print('No stack trace available for thread (%d)' % tid)
    except WindowsError:
        print("Can't get stack trace for thread (%d)" % tid)