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
def do_ba(self, arg):
    """
        [~thread] ba <a|w|e> <1|2|4|8> <address> - set hardware breakpoint
        """
    debug = self.debug
    thread = self.get_thread_from_prefix()
    pid = thread.get_pid()
    tid = thread.get_tid()
    if not debug.is_debugee(pid):
        raise CmdError('target thread is not being debugged')
    token_list = self.split_tokens(arg, 3, 3)
    access = token_list[0].lower()
    size = token_list[1]
    address = token_list[2]
    if access == 'a':
        access = debug.BP_BREAK_ON_ACCESS
    elif access == 'w':
        access = debug.BP_BREAK_ON_WRITE
    elif access == 'e':
        access = debug.BP_BREAK_ON_EXECUTION
    else:
        raise CmdError('bad access type: %s' % token_list[0])
    if size == '1':
        size = debug.BP_WATCH_BYTE
    elif size == '2':
        size = debug.BP_WATCH_WORD
    elif size == '4':
        size = debug.BP_WATCH_DWORD
    elif size == '8':
        size = debug.BP_WATCH_QWORD
    else:
        raise CmdError('bad breakpoint size: %s' % size)
    thread = self.get_thread_from_prefix()
    tid = thread.get_tid()
    pid = thread.get_pid()
    if not debug.is_debugee(pid):
        raise CmdError('target process is not being debugged')
    address = self.input_address(address, pid)
    if debug.has_hardware_breakpoint(tid, address):
        debug.erase_hardware_breakpoint(tid, address)
    debug.define_hardware_breakpoint(tid, address, access, size)
    debug.enable_hardware_breakpoint(tid, address)