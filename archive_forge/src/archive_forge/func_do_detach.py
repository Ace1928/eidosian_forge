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
def do_detach(self, arg):
    """
        [~process] detach - detach from the current process
        detach - detach from the current process
        detach <target> [target...] - detach from the given process(es)
        """
    debug = self.debug
    token_list = self.split_tokens(arg)
    if self.cmdprefix:
        token_list.insert(0, self.cmdprefix)
    targets = self.input_process_list(token_list)
    if not targets:
        if self.lastEvent is None:
            raise CmdError('no current process set')
        targets = [self.lastEvent.get_pid()]
    for pid in targets:
        try:
            debug.detach(pid)
            print('Detached from process (%d)' % pid)
        except Exception:
            print("Error: can't detach from process (%d)" % pid)