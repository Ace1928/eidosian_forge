from __future__ import with_statement
from winappdbg import win32
from winappdbg.registry import Registry
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses, DebugRegister, \
from winappdbg.process import _ProcessContainer
from winappdbg.window import Window
import sys
import os
import ctypes
import warnings
from os import path, getenv
@staticmethod
def read_msr(address):
    """
        Read the contents of the specified MSR (Machine Specific Register).

        @type  address: int
        @param address: MSR to read.

        @rtype:  int
        @return: Value of the specified MSR.

        @raise WindowsError:
            Raises an exception on error.

        @raise NotImplementedError:
            Current architecture is not C{i386} or C{amd64}.

        @warning:
            It could potentially brick your machine.
            It works on my machine, but your mileage may vary.
        """
    if win32.arch not in (win32.ARCH_I386, win32.ARCH_AMD64):
        raise NotImplementedError('MSR reading is only supported on i386 or amd64 processors.')
    msr = win32.SYSDBG_MSR()
    msr.Address = address
    msr.Data = 0
    win32.NtSystemDebugControl(win32.SysDbgReadMsr, InputBuffer=msr, OutputBuffer=msr)
    return msr.Data