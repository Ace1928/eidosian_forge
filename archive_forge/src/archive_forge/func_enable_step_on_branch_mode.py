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
@classmethod
def enable_step_on_branch_mode(cls):
    """
        When tracing, call this on every single step event
        for step on branch mode.

        @raise WindowsError:
            Raises C{ERROR_DEBUGGER_INACTIVE} if the debugger is not attached
            to least one process.

        @raise NotImplementedError:
            Current architecture is not C{i386} or C{amd64}.

        @warning:
            This method uses the processor's machine specific registers (MSR).
            It could potentially brick your machine.
            It works on my machine, but your mileage may vary.

        @note:
            It doesn't seem to work in VMWare or VirtualBox machines.
            Maybe it fails in other virtualization/emulation environments,
            no extensive testing was made so far.
        """
    cls.write_msr(DebugRegister.DebugCtlMSR, DebugRegister.BranchTrapFlag | DebugRegister.LastBranchRecord)