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
def input_address(self, token, pid=None, tid=None):
    address = None
    if self.is_register(token):
        if tid is None:
            if self.lastEvent is None or pid != self.lastEvent.get_pid():
                msg = "can't resolve register (%s) for unknown thread"
                raise CmdError(msg % token)
            tid = self.lastEvent.get_tid()
        address = self.input_register(token, tid)
    if address is None:
        try:
            address = self.input_hexadecimal_integer(token)
        except ValueError:
            if pid is None:
                if self.lastEvent is None:
                    raise CmdError('no current process set')
                process = self.lastEvent.get_process()
            elif self.lastEvent is not None and pid == self.lastEvent.get_pid():
                process = self.lastEvent.get_process()
            else:
                try:
                    process = self.debug.system.get_process(pid)
                except KeyError:
                    raise CmdError('process not found (%d)' % pid)
            try:
                address = process.resolve_label(token)
            except Exception:
                raise CmdError('unknown address (%s)' % token)
    return address