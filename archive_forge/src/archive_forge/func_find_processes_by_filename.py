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
def find_processes_by_filename(self, fileName):
    """
        @type  fileName: str
        @param fileName: Filename to search for.
            If it's a full pathname, the match must be exact.
            If it's a base filename only, the file part is matched,
            regardless of the directory where it's located.

        @note: If the process is not found and the file extension is not
            given, this method will search again assuming a default
            extension (.exe).

        @rtype:  list of tuple( L{Process}, str )
        @return: List of processes matching the given main module filename.
            Each tuple contains a Process object and it's filename.
        """
    found = self.__find_processes_by_filename(fileName)
    if not found:
        fn, ext = PathOperations.split_extension(fileName)
        if not ext:
            fileName = '%s.exe' % fn
            found = self.__find_processes_by_filename(fileName)
    return found