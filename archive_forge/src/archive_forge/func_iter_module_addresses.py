from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def iter_module_addresses(self):
    """
        @see:    L{iter_modules}
        @rtype:  dictionary-keyiterator
        @return: Iterator of DLL base addresses in this snapshot.
        """
    self.__initialize_snapshot()
    return compat.iterkeys(self.__moduleDict)