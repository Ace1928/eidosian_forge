from winappdbg import win32
from winappdbg import compat
from winappdbg.system import System
from winappdbg.textio import HexDump, CrashDump
from winappdbg.util import StaticClass, MemoryAddresses, PathOperations
import sys
import os
import time
import zlib
import warnings
def unmarshall_value(self, value):
    """
        Unmarshalls a Crash object read from the database.

        @type  value: str
        @param value: Object to convert.

        @rtype:  L{Crash}
        @return: Converted object.
        """
    value = str(value)
    if self.escapeValues:
        value = value.decode('hex')
    if self.compressValues:
        value = zlib.decompress(value)
    value = pickle.loads(value)
    return value