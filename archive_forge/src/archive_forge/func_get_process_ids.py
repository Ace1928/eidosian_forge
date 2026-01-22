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
def get_process_ids(self):
    """
        @see:    L{iter_process_ids}
        @rtype:  list( int )
        @return: List of global process IDs in this snapshot.
        """
    self.__initialize_snapshot()
    return compat.keys(self.__processDict)