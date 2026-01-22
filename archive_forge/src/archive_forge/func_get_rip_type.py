from winappdbg import win32
from winappdbg import compat
from winappdbg.win32 import FileHandle, ProcessHandle, ThreadHandle
from winappdbg.breakpoint import ApiHook
from winappdbg.module import Module
from winappdbg.thread import Thread
from winappdbg.process import Process
from winappdbg.textio import HexDump
from winappdbg.util import StaticClass, PathOperations
import sys
import ctypes
import warnings
import traceback
def get_rip_type(self):
    """
        @rtype:  int
        @return: RIP type code as defined by the Win32 API.
            May be C{0} or one of the following:
             - L{win32.SLE_ERROR}
             - L{win32.SLE_MINORERROR}
             - L{win32.SLE_WARNING}
        """
    return self.raw.u.RipInfo.dwType