from winappdbg import win32
from winappdbg import compat
from winappdbg.win32 import FileHandle, ProcessHandle, ThreadHandle
from winappdbg.breakpoint import ApiHook
from winappdbg.module import Module
from winappdbg.thread import Thread
from winappdbg.process import Process
from winappdbg.textio import HexDump
from winappdbg.util import StaticClass, PathOperations
import sys
import ctypes
import warnings
import traceback
def get_debug_string(self):
    """
        @rtype:  str, compat.unicode
        @return: String sent by the debugee.
            It may be ANSI or Unicode and may end with a null character.
        """
    return self.get_process().peek_string(self.raw.u.DebugString.lpDebugStringData, bool(self.raw.u.DebugString.fUnicode), self.raw.u.DebugString.nDebugStringLength)