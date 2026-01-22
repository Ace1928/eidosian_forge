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
def get_peb_address(self):
    """
        Returns a remote pointer to the PEB.

        @rtype:  int
        @return: Remote pointer to the L{win32.PEB} structure.
            Returns C{None} on error.
        """
    try:
        return self._peb_ptr
    except AttributeError:
        hProcess = self.get_handle(win32.PROCESS_QUERY_INFORMATION)
        pbi = win32.NtQueryInformationProcess(hProcess, win32.ProcessBasicInformation)
        address = pbi.PebBaseAddress
        self._peb_ptr = address
        return address