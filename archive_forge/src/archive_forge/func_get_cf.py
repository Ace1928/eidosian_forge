from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_cf(self):
    """
            @rtype:  bool
            @return: Boolean value of the Carry flag.
            """
    return self.get_flag_value(self.Flags.Carry)