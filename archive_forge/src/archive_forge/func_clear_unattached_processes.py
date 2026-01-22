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
def clear_unattached_processes(self):
    """
        Removes Process objects from the snapshot
        referring to processes not being debugged.
        """
    for pid in self.get_process_ids():
        aProcess = self.get_process(pid)
        if not aProcess.is_being_debugged():
            self._del_process(aProcess)