from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def dont_watch_buffer(self, bw, *argv, **argd):
    """
        Clears a page breakpoint set by L{watch_buffer}.

        @type  bw: L{BufferWatch}
        @param bw:
            Buffer watch identifier returned by L{watch_buffer}.
        """
    if not (argv or argd):
        self.__clear_buffer_watch(bw)
    else:
        argv = list(argv)
        argv.insert(0, bw)
        if 'pid' in argd:
            argv.insert(0, argd.pop('pid'))
        if 'address' in argd:
            argv.insert(1, argd.pop('address'))
        if 'size' in argd:
            argv.insert(2, argd.pop('size'))
        if argd:
            raise TypeError('Wrong arguments for dont_watch_buffer()')
        try:
            pid, address, size = argv
        except ValueError:
            raise TypeError('Wrong arguments for dont_watch_buffer()')
        self.__clear_buffer_watch_old_method(pid, address, size)