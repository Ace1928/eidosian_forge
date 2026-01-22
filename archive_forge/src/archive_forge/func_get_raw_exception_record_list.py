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
def get_raw_exception_record_list(self):
    """
        Traverses the exception record linked list and builds a Python list.

        Nested exception records are received for nested exceptions. This
        happens when an exception is raised in the debugee while trying to
        handle a previous exception.

        @rtype:  list( L{win32.EXCEPTION_RECORD} )
        @return:
            List of raw exception record structures as used by the Win32 API.

            There is always at least one exception record, so the list is
            never empty. All other methods of this class read from the first
            exception record only, that is, the most recent exception.
        """
    nested = list()
    record = self.raw.u.Exception
    while True:
        record = record.ExceptionRecord
        if not record:
            break
        nested.append(record)
    return nested