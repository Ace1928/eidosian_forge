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
def get_peb(self):
    """
        Returns a copy of the PEB.
        To dereference pointers in it call L{Process.read_structure}.

        @rtype:  L{win32.PEB}
        @return: PEB structure.
        @raise WindowsError: An exception is raised on error.
        """
    self.get_handle(win32.PROCESS_VM_READ | win32.PROCESS_QUERY_INFORMATION)
    return self.read_structure(self.get_peb_address(), win32.PEB)