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
def read_float(self, lpBaseAddress):
    """
        Reads a float from the memory of the process.

        @see: L{peek_float}

        @type  lpBaseAddress: int
        @param lpBaseAddress: Memory address to begin reading.

        @rtype:  int
        @return: Floating point value read from the process memory.

        @raise WindowsError: On error an exception is raised.
        """
    return self.__read_c_type(lpBaseAddress, '@f', ctypes.c_float)