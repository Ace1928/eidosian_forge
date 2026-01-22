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
def input_address_range(self, token_list, pid=None, tid=None):
    if len(token_list) == 2:
        token_1, token_2 = token_list
        address = self.input_address(token_1, pid, tid)
        try:
            size = self.input_integer(token_2)
        except ValueError:
            raise CmdError('bad address range: %s %s' % (token_1, token_2))
    elif len(token_list) == 1:
        token = token_list[0]
        if '-' in token:
            try:
                token_1, token_2 = token.split('-')
            except Exception:
                raise CmdError('bad address range: %s' % token)
            address = self.input_address(token_1, pid, tid)
            size = self.input_address(token_2, pid, tid) - address
        else:
            address = self.input_address(token, pid, tid)
            size = None
    return (address, size)