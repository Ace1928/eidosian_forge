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
def search_text(self, text, encoding='utf-16le', caseSensitive=False, minAddr=None, maxAddr=None):
    """
        Search for the given text within the process memory.

        @type  text: str or compat.unicode
        @param text: Text to search for.

        @type  encoding: str
        @param encoding: (Optional) Encoding for the text parameter.
            Only used when the text to search for is a Unicode string.
            Don't change unless you know what you're doing!

        @type  caseSensitive: bool
        @param caseSensitive: C{True} of the search is case sensitive,
            C{False} otherwise.

        @type  minAddr: int
        @param minAddr: (Optional) Start the search at this memory address.

        @type  maxAddr: int
        @param maxAddr: (Optional) Stop the search at this memory address.

        @rtype:  iterator of tuple( int, str )
        @return: An iterator of tuples. Each tuple contains the following:
             - The memory address where the pattern was found.
             - The text that matches the pattern.

        @raise WindowsError: An error occurred when querying or reading the
            process memory.
        """
    pattern = TextPattern(text, encoding, caseSensitive)
    matches = Search.search_process(self, pattern, minAddr, maxAddr)
    for addr, size, data in matches:
        yield (addr, data)