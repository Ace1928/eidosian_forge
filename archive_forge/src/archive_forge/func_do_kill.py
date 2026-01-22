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
def do_kill(self, arg):
    """
        [~process] kill - kill a process
        [~thread] kill - kill a thread
        kill - kill the current process
        kill * - kill all debugged processes
        kill <processes and/or threads...> - kill the given processes and threads
        """
    if arg:
        if arg == '*':
            target_pids = self.debug.get_debugee_pids()
            target_tids = list()
        else:
            target_pids = set()
            target_tids = set()
            if self.cmdprefix:
                pid, tid = self.get_process_and_thread_ids_from_prefix()
                if tid is None:
                    target_tids.add(tid)
                else:
                    target_pids.add(pid)
            for token in self.split_tokens(arg):
                try:
                    pid = self.input_process(token)
                    target_pids.add(pid)
                except CmdError:
                    try:
                        tid = self.input_process(token)
                        target_pids.add(pid)
                    except CmdError:
                        msg = 'unknown process or thread (%s)' % token
                        raise CmdError(msg)
            target_pids = list(target_pids)
            target_tids = list(target_tids)
            target_pids.sort()
            target_tids.sort()
        msg = 'You are about to kill %d processes and %d threads.'
        msg = msg % (len(target_pids), len(target_tids))
        if self.ask_user(msg):
            for pid in target_pids:
                self.kill_process(pid)
            for tid in target_tids:
                self.kill_thread(tid)
    elif self.cmdprefix:
        pid, tid = self.get_process_and_thread_ids_from_prefix()
        if tid is None:
            if self.lastEvent is not None and pid == self.lastEvent.get_pid():
                msg = 'You are about to kill the current process.'
            else:
                msg = 'You are about to kill process %d.' % pid
            if self.ask_user(msg):
                self.kill_process(pid)
        else:
            if self.lastEvent is not None and tid == self.lastEvent.get_tid():
                msg = 'You are about to kill the current thread.'
            else:
                msg = 'You are about to kill thread %d.' % tid
            if self.ask_user(msg):
                self.kill_thread(tid)
    else:
        if self.lastEvent is None:
            raise CmdError('no current process set')
        pid = self.lastEvent.get_pid()
        if self.ask_user('You are about to kill the current process.'):
            self.kill_process(pid)