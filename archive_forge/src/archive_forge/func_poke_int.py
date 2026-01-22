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
def poke_int(self, lpBaseAddress, unpackedValue):
    """
        Writes a signed integer to the memory of the process.

        @note: Page permissions may be changed temporarily while writing.

        @see: L{write_int}

        @type  lpBaseAddress: int
        @param lpBaseAddress: Memory address to begin writing.

        @type  unpackedValue: int, long
        @param unpackedValue: Value to write.

        @rtype:  int
        @return: Number of bytes written.
            May be less than the number of bytes to write.
        """
    return self.__poke_c_type(lpBaseAddress, '@l', unpackedValue)