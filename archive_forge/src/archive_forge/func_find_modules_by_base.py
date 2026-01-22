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
def find_modules_by_base(self, lpBaseOfDll):
    """
        @rtype:  list( L{Module}... )
        @return: List of Module objects with the given base address.
        """
    found = list()
    for aProcess in self.iter_processes():
        if aProcess.has_module(lpBaseOfDll):
            aModule = aProcess.get_module(lpBaseOfDll)
            found.append((aProcess, aModule))
    return found