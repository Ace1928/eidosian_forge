from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_bits(self):
    """
        @rtype:  str
        @return: The number of bits in which this thread believes to be
            running. For example, if running a 32 bit binary in a 64 bit
            machine, the number of bits returned by this method will be C{32},
            but the value of L{System.arch} will be C{64}.
        """
    if win32.bits == 32 and (not win32.wow64):
        return 32
    return self.get_process().get_bits()