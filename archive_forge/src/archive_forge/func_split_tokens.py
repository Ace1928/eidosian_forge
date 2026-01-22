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
def split_tokens(self, arg, min_count=0, max_count=None):
    token_list = self.debug.system.cmdline_to_argv(arg)
    if len(token_list) < min_count:
        raise CmdError('missing parameters.')
    if max_count and len(token_list) > max_count:
        raise CmdError('too many parameters.')
    return token_list