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
def get_services(self):
    """
        Retrieves the list of system services that are currently running in
        this process.

        @see: L{System.get_services}

        @rtype:  list( L{win32.ServiceStatusProcessEntry} )
        @return: List of service status descriptors.
        """
    self.__load_System_class()
    pid = self.get_pid()
    return [d for d in System.get_active_services() if d.ProcessId == pid]