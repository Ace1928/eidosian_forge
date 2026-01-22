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
def is_first_chance(self):
    """
        @rtype:  bool
        @return: C{True} for first chance exceptions, C{False} for last chance.
        """
    return self.raw.u.Exception.dwFirstChance != 0