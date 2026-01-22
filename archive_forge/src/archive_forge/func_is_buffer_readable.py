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
def is_buffer_readable(self, address, size):
    """
        Determines if the given memory area is readable.

        @note: Returns always C{False} for kernel mode addresses.

        @see: L{mquery}

        @type  address: int
        @param address: Memory address.

        @type  size: int
        @param size: Number of bytes. Must be greater than zero.

        @rtype:  bool
        @return: C{True} if the memory area is readable, C{False} otherwise.

        @raise ValueError: The size argument must be greater than zero.
        @raise WindowsError: On error an exception is raised.
        """
    if size <= 0:
        raise ValueError('The size argument must be greater than zero')
    while size > 0:
        try:
            mbi = self.mquery(address)
        except WindowsError:
            e = sys.exc_info()[1]
            if e.winerror == win32.ERROR_INVALID_PARAMETER:
                return False
            raise
        if not mbi.is_readable():
            return False
        size = size - mbi.RegionSize
    return True