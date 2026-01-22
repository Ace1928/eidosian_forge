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
def get_exception_information_as_list(self):
    """
        @rtype:  list( int )
        @return: Exception information block.
        """
    info = self.raw.u.Exception.ExceptionRecord.ExceptionInformation
    data = list()
    for index in compat.xrange(0, win32.EXCEPTION_MAXIMUM_PARAMETERS):
        value = info[index]
        if value is None:
            value = 0
        data.append(value)
    return data