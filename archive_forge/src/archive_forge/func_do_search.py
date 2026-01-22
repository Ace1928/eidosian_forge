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
def do_search(self, arg):
    """
        [~process] s [address-address] <search string>
        [~process] search [address-address] <search string>
        """
    token_list = self.split_tokens(arg, 1, 3)
    pid, tid = self.get_process_and_thread_ids_from_prefix()
    process = self.get_process(pid)
    if len(token_list) == 1:
        pattern = token_list[0]
        minAddr = None
        maxAddr = None
    else:
        pattern = token_list[-1]
        addr, size = self.input_address_range(token_list[:-1], pid, tid)
        minAddr = addr
        maxAddr = addr + size
    iter = process.search_bytes(pattern)
    if process.get_bits() == 32:
        addr_width = 8
    else:
        addr_width = 16
    for addr in iter:
        print(HexDump.address(addr, addr_width))