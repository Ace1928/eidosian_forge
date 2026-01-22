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
def do_disassemble(self, arg):
    """
        [~thread] u [register] - show code disassembly
        [~process] u [address] - show code disassembly
        [~thread] disassemble [register] - show code disassembly
        [~process] disassemble [address] - show code disassembly
        """
    if not arg:
        arg = self.default_disasm_target
    token_list = self.split_tokens(arg, 1, 1)
    pid, tid = self.get_process_and_thread_ids_from_prefix()
    process = self.get_process(pid)
    address = self.input_address(token_list[0], pid, tid)
    try:
        code = process.disassemble(address, 15 * 8)[:8]
    except Exception:
        msg = "can't disassemble address %s"
        msg = msg % HexDump.address(address)
        raise CmdError(msg)
    if code:
        label = process.get_label_at_address(address)
        last_code = code[-1]
        next_address = last_code[0] + last_code[1]
        next_address = HexOutput.integer(next_address)
        self.default_disasm_target = next_address
        print('%s:' % label)
        for line in code:
            print(CrashDump.dump_code_line(line, bShowDump=False))