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
def get_file_handle(self):
    """
        @rtype:  None or L{FileHandle}
        @return: File handle to the recently unloaded DLL.
            Returns C{None} if the handle is not available.
        """
    hFile = self.get_module().hFile
    if hFile in (0, win32.NULL, win32.INVALID_HANDLE_VALUE):
        hFile = None
    return hFile