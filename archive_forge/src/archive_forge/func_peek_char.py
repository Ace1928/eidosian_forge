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
def peek_char(self, lpBaseAddress):
    """
        Reads a single character from the memory of the process.

        @see: L{read_char}

        @type  lpBaseAddress: int
        @param lpBaseAddress: Memory address to begin reading.

        @rtype:  int
        @return: Character read from the process memory.
            Returns zero on error.
        """
    char = self.peek(lpBaseAddress, 1)
    if char:
        return ord(char)
    return 0