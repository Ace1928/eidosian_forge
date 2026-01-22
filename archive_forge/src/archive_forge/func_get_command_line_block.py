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
def get_command_line_block(self):
    """
        Retrieves the command line block memory address and size.

        @rtype:  tuple(int, int)
        @return: Tuple with the memory address of the command line block
            and it's maximum size in Unicode characters.

        @raise WindowsError: On error an exception is raised.
        """
    peb = self.get_peb()
    pp = self.read_structure(peb.ProcessParameters, win32.RTL_USER_PROCESS_PARAMETERS)
    s = pp.CommandLine
    return (s.Buffer, s.MaximumLength)