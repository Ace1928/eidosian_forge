from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_hardware_breakpoint(self, dwThreadId, address):
    """
        Returns the internally used breakpoint object,
        for the code breakpoint defined at the given address.

        @warning: It's usually best to call the L{Debug} methods
            instead of accessing the breakpoint objects directly.

        @see:
            L{define_hardware_breakpoint},
            L{has_hardware_breakpoint},
            L{get_code_breakpoint},
            L{enable_hardware_breakpoint},
            L{enable_one_shot_hardware_breakpoint},
            L{disable_hardware_breakpoint},
            L{erase_hardware_breakpoint}

        @type  dwThreadId: int
        @param dwThreadId: Thread global ID.

        @type  address: int
        @param address: Memory address where the breakpoint is defined.

        @rtype:  L{HardwareBreakpoint}
        @return: The hardware breakpoint object.
        """
    if dwThreadId not in self.__hardwareBP:
        msg = 'No hardware breakpoints set for thread %d'
        raise KeyError(msg % dwThreadId)
    for bp in self.__hardwareBP[dwThreadId]:
        if bp.is_here(address):
            return bp
    msg = 'No hardware breakpoint at thread %d, address %s'
    raise KeyError(msg % (dwThreadId, HexDump.address(address)))