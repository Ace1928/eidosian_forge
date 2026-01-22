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
def get_environment_block(self):
    """
        Retrieves the environment block memory address for the process.

        @note: The size is always enough to contain the environment data, but
            it may not be an exact size. It's best to read the memory and
            scan for two null wide chars to find the actual size.

        @rtype:  tuple(int, int)
        @return: Tuple with the memory address of the environment block
            and it's size.

        @raise WindowsError: On error an exception is raised.
        """
    peb = self.get_peb()
    pp = self.read_structure(peb.ProcessParameters, win32.RTL_USER_PROCESS_PARAMETERS)
    Environment = pp.Environment
    try:
        EnvironmentSize = pp.EnvironmentSize
    except AttributeError:
        mbi = self.mquery(Environment)
        EnvironmentSize = mbi.RegionSize + mbi.BaseAddress - Environment
    return (Environment, EnvironmentSize)