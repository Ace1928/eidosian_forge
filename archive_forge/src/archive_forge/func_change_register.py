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
def change_register(self, register, value, tid=None):
    if tid is None:
        if self.lastEvent is None:
            raise CmdError('no current process set')
        thread = self.lastEvent.get_thread()
    else:
        try:
            thread = self.debug.system.get_thread(tid)
        except KeyError:
            raise CmdError('thread not found (%d)' % tid)
    try:
        value = self.input_integer(value)
    except ValueError:
        pid = thread.get_pid()
        value = self.input_address(value, pid, tid)
    thread.suspend()
    try:
        ctx = thread.get_context()
        register = register.lower()
        if register in self.register_names:
            register = register.title()
        if register in self.segment_names:
            register = 'Seg%s' % register.title()
            value = value & 65535
        if register in self.register_alias_16:
            register = self.register_alias_16[register]
            previous = ctx.get(register) & 4294901760
            value = value & 65535 | previous
        if register in self.register_alias_8_low:
            register = self.register_alias_8_low[register]
            previous = ctx.get(register) % 4294967040
            value = value & 255 | previous
        if register in self.register_alias_8_high:
            register = self.register_alias_8_high[register]
            previous = ctx.get(register) % 4294902015
            value = (value & 255) << 8 | previous
        ctx.__setitem__(register, value)
        thread.set_context(ctx)
    finally:
        thread.resume()