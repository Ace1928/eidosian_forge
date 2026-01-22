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
def do_find(self, arg):
    """
        [~process] f <string> - find the string in the process memory
        [~process] find <string> - find the string in the process memory
        """
    if not arg:
        raise CmdError('missing parameter: string')
    process = self.get_process_from_prefix()
    self.find_in_memory(arg, process)