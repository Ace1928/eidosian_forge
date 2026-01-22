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
def search_hexa(self, hexa, minAddr=None, maxAddr=None):
    """
        Search for the given hexadecimal pattern within the process memory.

        Hex patterns must be in this form::
            "68 65 6c 6c 6f 20 77 6f 72 6c 64"  # "hello world"

        Spaces are optional. Capitalization of hex digits doesn't matter.
        This is exactly equivalent to the previous example::
            "68656C6C6F20776F726C64"            # "hello world"

        Wildcards are allowed, in the form of a C{?} sign in any hex digit::
            "5? 5? c3"          # pop register / pop register / ret
            "b8 ?? ?? ?? ??"    # mov eax, immediate value

        @type  hexa: str
        @param hexa: Pattern to search for.

        @type  minAddr: int
        @param minAddr: (Optional) Start the search at this memory address.

        @type  maxAddr: int
        @param maxAddr: (Optional) Stop the search at this memory address.

        @rtype:  iterator of tuple( int, str )
        @return: An iterator of tuples. Each tuple contains the following:
             - The memory address where the pattern was found.
             - The bytes that match the pattern.

        @raise WindowsError: An error occurred when querying or reading the
            process memory.
        """
    pattern = HexPattern(hexa)
    matches = Search.search_process(self, pattern, minAddr, maxAddr)
    for addr, size, data in matches:
        yield (addr, data)