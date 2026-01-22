from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_windows(self):
    """
        @rtype:  list of L{Window}
        @return: Returns a list of windows handled by this process.
        """
    window_list = list()
    for thread in self.iter_threads():
        window_list.extend(thread.get_windows())
    return window_list