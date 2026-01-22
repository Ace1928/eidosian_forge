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
def poke(self, lpBaseAddress, lpBuffer):
    """
        Writes to the memory of the process.

        @note: Page permissions may be changed temporarily while writing.

        @see: L{write}

        @type  lpBaseAddress: int
        @param lpBaseAddress: Memory address to begin writing.

        @type  lpBuffer: str
        @param lpBuffer: Bytes to write.

        @rtype:  int
        @return: Number of bytes written.
            May be less than the number of bytes to write.
        """
    assert isinstance(lpBuffer, compat.bytes)
    hProcess = self.get_handle(win32.PROCESS_VM_WRITE | win32.PROCESS_VM_OPERATION | win32.PROCESS_QUERY_INFORMATION)
    mbi = self.mquery(lpBaseAddress)
    if not mbi.has_content():
        raise ctypes.WinError(win32.ERROR_INVALID_ADDRESS)
    if mbi.is_image() or mbi.is_mapped():
        prot = win32.PAGE_WRITECOPY
    elif mbi.is_writeable():
        prot = None
    elif mbi.is_executable():
        prot = win32.PAGE_EXECUTE_READWRITE
    else:
        prot = win32.PAGE_READWRITE
    if prot is not None:
        try:
            self.mprotect(lpBaseAddress, len(lpBuffer), prot)
        except Exception:
            prot = None
            msg = 'Failed to adjust page permissions for process %s at address %s: %s'
            msg = msg % (self.get_pid(), HexDump.address(lpBaseAddress, self.get_bits()), traceback.format_exc())
            warnings.warn(msg, RuntimeWarning)
    try:
        r = win32.WriteProcessMemory(hProcess, lpBaseAddress, lpBuffer)
    finally:
        if prot is not None:
            self.mprotect(lpBaseAddress, len(lpBuffer), mbi.Protect)
    return r