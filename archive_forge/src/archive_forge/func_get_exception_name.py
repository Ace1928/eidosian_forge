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
def get_exception_name(self):
    """
        @rtype:  str
        @return: Name of the exception as defined by the Win32 API.
        """
    code = self.get_exception_code()
    unk = HexDump.integer(code)
    return self.__exceptionName.get(code, unk)