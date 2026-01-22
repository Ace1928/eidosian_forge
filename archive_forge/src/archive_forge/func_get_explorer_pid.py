from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump, HexInput
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses
from winappdbg.module import Module, _ModuleContainer
from winappdbg.thread import Thread, _ThreadContainer
from winappdbg.window import Window
from winappdbg.search import Search, \
from winappdbg.disasm import Disassembler
import re
import os
import os.path
import ctypes
import struct
import warnings
import traceback
def get_explorer_pid(self):
    """
        Tries to find the process ID for "explorer.exe".

        @rtype:  int or None
        @return: Returns the process ID, or C{None} on error.
        """
    try:
        exp = win32.SHGetFolderPath(win32.CSIDL_WINDOWS)
    except Exception:
        exp = None
    if not exp:
        exp = os.getenv('SystemRoot')
    if exp:
        exp = os.path.join(exp, 'explorer.exe')
        exp_list = self.find_processes_by_filename(exp)
        if exp_list:
            return exp_list[0][0].get_pid()
    return None