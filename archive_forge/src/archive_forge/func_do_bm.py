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
def do_bm(self, arg):
    """
        [~process] bm <address-address> - set memory breakpoint
        """
    pid = self.get_process_id_from_prefix()
    if not self.debug.is_debugee(pid):
        raise CmdError('target process is not being debugged')
    process = self.get_process(pid)
    token_list = self.split_tokens(arg, 1, 2)
    address, size = self.input_address_range(token_list[0], pid)
    self.debug.watch_buffer(pid, address, size)