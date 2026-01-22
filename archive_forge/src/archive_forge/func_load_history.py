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
def load_history(self):
    global readline
    if readline is None:
        try:
            import readline
        except ImportError:
            return
    if self.history_file_full_path is None:
        folder = os.environ.get('USERPROFILE', '')
        if not folder:
            folder = os.environ.get('HOME', '')
        if not folder:
            folder = os.path.split(sys.argv[0])[1]
        if not folder:
            folder = os.path.curdir
        self.history_file_full_path = os.path.join(folder, self.history_file)
    try:
        if os.path.exists(self.history_file_full_path):
            readline.read_history_file(self.history_file_full_path)
    except IOError:
        e = sys.exc_info()[1]
        warnings.warn('Cannot load history file, reason: %s' % str(e))