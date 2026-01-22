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
def input_full_address_range(self, token_list):
    pid, tid = self.get_process_and_thread_ids_from_prefix()
    address, size = self.input_address_range(token_list, pid, tid)
    return (pid, tid, address, size)