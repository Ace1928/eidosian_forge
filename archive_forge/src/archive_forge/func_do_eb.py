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
def do_eb(self, arg):
    """
        [~process] eb <address> <data> - write the data to the specified address
        """
    pid = self.get_process_id_from_prefix()
    token_list = self.split_tokens(arg, 2)
    address = self.input_address(token_list[0], pid)
    data = HexInput.hexadecimal(' '.join(token_list[1:]))
    self.write_memory(address, data, pid)