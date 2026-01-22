from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def dont_stalk_buffer(self, bw, *argv, **argd):
    """
        Clears a page breakpoint set by L{stalk_buffer}.

        @type  bw: L{BufferWatch}
        @param bw:
            Buffer watch identifier returned by L{stalk_buffer}.
        """
    self.dont_watch_buffer(bw, *argv, **argd)