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
def print_event(self, event):
    code = HexDump.integer(event.get_event_code())
    name = event.get_event_name()
    desc = event.get_event_description()
    if code in desc:
        print('')
        print('%s: %s' % (name, desc))
    else:
        print('')
        print('%s (%s): %s' % (name, code, desc))
    self.print_event_location(event)