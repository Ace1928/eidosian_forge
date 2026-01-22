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
def peek_string(self, lpBaseAddress, fUnicode=False, dwMaxSize=4096):
    """
        Tries to read an ASCII or Unicode string
        from the address space of the process.

        @see: L{read_string}

        @type  lpBaseAddress: int
        @param lpBaseAddress: Memory address to begin reading.

        @type  fUnicode: bool
        @param fUnicode: C{True} is the string is expected to be Unicode,
            C{False} if it's expected to be ANSI.

        @type  dwMaxSize: int
        @param dwMaxSize: Maximum allowed string length to read, in bytes.

        @rtype:  str, compat.unicode
        @return: String read from the process memory space.
            It B{doesn't} include the terminating null character.
            Returns an empty string on failure.
        """
    if not lpBaseAddress or dwMaxSize == 0:
        if fUnicode:
            return u''
        return ''
    if not dwMaxSize:
        dwMaxSize = 4096
    szString = self.peek(lpBaseAddress, dwMaxSize)
    if fUnicode:
        szString = compat.unicode(szString, 'U16', 'replace')
        szString = szString[:szString.find(u'\x00')]
    else:
        szString = szString[:szString.find('\x00')]
    return szString