from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def set_process(self, process=None):
    """
        Manually set the parent Process object. Use with care!

        @type  process: L{Process}
        @param process: (Optional) Process object. Use C{None} for no process.
        """
    if process is None:
        self.dwProcessId = None
        self.__process = None
    else:
        self.__load_Process_class()
        if not isinstance(process, Process):
            msg = 'Parent process must be a Process instance, '
            msg += 'got %s instead' % type(process)
            raise TypeError(msg)
        self.dwProcessId = process.get_pid()
        self.__process = process