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
def do_bp(self, arg):
    """
        [~process] bp <address> - set a code breakpoint
        """
    pid = self.get_process_id_from_prefix()
    if not self.debug.is_debugee(pid):
        raise CmdError('target process is not being debugged')
    process = self.get_process(pid)
    token_list = self.split_tokens(arg, 1, 1)
    try:
        address = self.input_address(token_list[0], pid)
        deferred = False
    except Exception:
        address = token_list[0]
        deferred = True
    if not address:
        address = token_list[0]
        deferred = True
    self.debug.break_at(pid, address)
    if deferred:
        print('Deferred breakpoint set at %s' % address)
    else:
        print('Breakpoint set at %s' % address)