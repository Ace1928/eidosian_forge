from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_seh_chain_pointer(self):
    """
        Get the pointer to the first structured exception handler block.

        @rtype:  int
        @return: Remote pointer to the first block of the structured exception
            handlers linked list. If the list is empty, the returned value is
            C{0xFFFFFFFF}.

        @raise NotImplementedError:
            This method is only supported in 32 bits versions of Windows.
        """
    if win32.arch != win32.ARCH_I386:
        raise NotImplementedError('SEH chain parsing is only supported in 32-bit Windows.')
    process = self.get_process()
    address = self.get_linear_address('SegFs', 0)
    return process.read_pointer(address)