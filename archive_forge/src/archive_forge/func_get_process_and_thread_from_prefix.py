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
def get_process_and_thread_from_prefix(self):
    pid, tid = self.get_process_and_thread_ids_from_prefix()
    process = self.get_process(pid)
    thread = self.get_thread(tid)
    return (process, thread)