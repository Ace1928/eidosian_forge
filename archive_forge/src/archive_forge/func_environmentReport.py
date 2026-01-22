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
def environmentReport(self):
    """
        @rtype: str
        @return: The process environment variables,
            merged and formatted for a report.
        """
    msg = ''
    if self.environment:
        for key, value in compat.iteritems(self.environment):
            msg += '  %s=%s\n' % (key, value)
    return msg