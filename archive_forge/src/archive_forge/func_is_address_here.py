from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def is_address_here(self, address):
    """
        Tries to determine if the given address belongs to this module.

        @type  address: int
        @param address: Memory address.

        @rtype:  bool or None
        @return: C{True} if the address belongs to the module,
            C{False} if it doesn't,
            and C{None} if it can't be determined.
        """
    base = self.get_base()
    size = self.get_size()
    if base and size:
        return base <= address < base + size
    return None