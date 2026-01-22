from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_process(self):
    """
        @rtype:  L{Process}
        @return: Parent Process object.
            Returns C{None} if unknown.
        """
    if self.__process is not None:
        return self.__process
    self.__load_Process_class()
    self.__process = Process(self.get_pid())
    return self.__process