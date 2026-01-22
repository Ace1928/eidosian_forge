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
def write_memory(self, address, data, pid=None):
    process = self.get_process(pid)
    try:
        process.write(address, data)
    except WindowsError:
        size = len(data)
        orig_address = HexOutput.integer(address)
        next_address = HexOutput.integer(address + size)
        msg = 'error reading process %d, from %s to %s (%d bytes)'
        msg = msg % (pid, orig_address, next_address, size)
        raise CmdError(msg)