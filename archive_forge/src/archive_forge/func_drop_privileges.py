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
def drop_privileges(cls, *privileges):
    """
        Drops privileges.

        @type  privileges: int...
        @param privileges: Privileges to drop.

        @raise WindowsError: Raises an exception on error.
        """
    cls.adjust_privileges(False, privileges)