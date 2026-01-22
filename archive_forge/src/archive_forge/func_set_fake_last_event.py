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
def set_fake_last_event(self, process):
    if self.lastEvent is None:
        self.debug.lastEvent = DummyEvent(self.debug)
        self.debug.lastEvent._process = process
        self.debug.lastEvent._thread = process.get_thread(process.get_thread_ids()[0])
        self.debug.lastEvent._pid = process.get_pid()
        self.debug.lastEvent._tid = self.lastEvent._thread.get_tid()