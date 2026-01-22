from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def read_stack_frame(self, structure, offset=0):
    """
        Reads the stack frame of the thread.

        @type  structure: ctypes.Structure
        @param structure: Structure of the stack frame.

        @type  offset: int
        @param offset: Offset from the frame pointer to begin reading.
            The frame pointer is the same returned by the L{get_fp} method.

        @rtype:  tuple
        @return: Tuple of elements read from the stack frame. The type of each
            element matches the types in the stack frame structure.
        """
    aProcess = self.get_process()
    stackData = aProcess.read_structure(self.get_fp() + offset, structure)
    return tuple([stackData.__getattribute__(name) for name, type in stackData._fields_])