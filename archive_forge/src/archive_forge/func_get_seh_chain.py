from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_seh_chain(self):
    """
        @rtype:  list of tuple( int, int )
        @return: List of structured exception handlers.
            Each SEH is represented as a tuple of two addresses:
                - Address of this SEH block
                - Address of the SEH callback function
            Do not confuse this with the contents of the SEH block itself,
            where the first member is a pointer to the B{next} block instead.

        @raise NotImplementedError:
            This method is only supported in 32 bits versions of Windows.
        """
    seh_chain = list()
    try:
        process = self.get_process()
        seh = self.get_seh_chain_pointer()
        while seh != 4294967295:
            seh_func = process.read_pointer(seh + 4)
            seh_chain.append((seh, seh_func))
            seh = process.read_pointer(seh)
    except WindowsError:
        seh_chain.append((seh, None))
    return seh_chain