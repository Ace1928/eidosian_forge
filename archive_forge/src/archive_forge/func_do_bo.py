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
def do_bo(self, arg):
    """
        [~process] bo <address> - make a code breakpoint one-shot
        [~thread] bo <address> - make a hardware breakpoint one-shot
        [~process] bo <address-address> - make a memory breakpoint one-shot
        [~process] bo <address> <size> - make a memory breakpoint one-shot
        """
    token_list = self.split_tokens(arg, 1, 2)
    pid, tid, address, size = self.input_breakpoint(token_list)
    debug = self.debug
    found = False
    if size is None:
        if tid is not None:
            if debug.has_hardware_breakpoint(tid, address):
                debug.enable_one_shot_hardware_breakpoint(tid, address)
                found = True
        if pid is not None:
            if debug.has_code_breakpoint(pid, address):
                debug.enable_one_shot_code_breakpoint(pid, address)
                found = True
    elif debug.has_page_breakpoint(pid, address):
        debug.enable_one_shot_page_breakpoint(pid, address)
        found = True
    if not found:
        print('Error: breakpoint not found.')