from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_label_at_pc(self):
    """
        @rtype:  str
        @return: Label that points to the instruction currently being executed.
        """
    return self.get_process().get_label_at_address(self.get_pc())