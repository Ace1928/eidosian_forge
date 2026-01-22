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
def mprotect(self, lpAddress, dwSize, flNewProtect):
    """
        Set memory protection in the address space of the process.

        @see: U{http://msdn.microsoft.com/en-us/library/aa366899.aspx}

        @type  lpAddress: int
        @param lpAddress: Address of memory to protect.

        @type  dwSize: int
        @param dwSize: Number of bytes to protect.

        @type  flNewProtect: int
        @param flNewProtect: New protect flags.

        @rtype:  int
        @return: Old protect flags.

        @raise WindowsError: On error an exception is raised.
        """
    hProcess = self.get_handle(win32.PROCESS_VM_OPERATION)
    return win32.VirtualProtectEx(hProcess, lpAddress, dwSize, flNewProtect)