from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def set_register(self, register, value):
    """
        Sets the value of a specific register.

        @type  register: str
        @param register: Register name.

        @rtype:  int
        @return: Register value.
        """
    context = self.get_context()
    context[register] = value
    self.set_context(context)