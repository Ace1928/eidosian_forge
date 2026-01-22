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
def input_process(self, token):
    targets = self.input_process_list([token])
    if len(targets) == 0:
        raise CmdError('missing process name or ID')
    if len(targets) > 1:
        msg = 'more than one process with that name:\n'
        for pid in targets:
            msg += '\t%d\n' % pid
        msg = msg[:-len('\n')]
        raise CmdError(msg)
    return targets[0]