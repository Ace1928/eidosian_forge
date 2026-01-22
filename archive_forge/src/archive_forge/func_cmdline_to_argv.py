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
@staticmethod
def cmdline_to_argv(lpCmdLine):
    """
        Convert a single command line string to a list of arguments.

        @type  lpCmdLine: str
        @param lpCmdLine: Command line string.
            The first token is the program to execute.

        @rtype:  list( str )
        @return: List of argument strings.
        """
    if not lpCmdLine:
        return []
    return win32.CommandLineToArgv(lpCmdLine)