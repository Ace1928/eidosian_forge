from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def read_stack_structure(self, structure, offset=0):
    """
        Reads the given structure at the top of the stack.

        @type  structure: ctypes.Structure
        @param structure: Structure of the data to read from the stack.

        @type  offset: int
        @param offset: Offset from the stack pointer to begin reading.
            The stack pointer is the same returned by the L{get_sp} method.

        @rtype:  tuple
        @return: Tuple of elements read from the stack. The type of each
            element matches the types in the stack frame structure.
        """
    aProcess = self.get_process()
    stackData = aProcess.read_structure(self.get_sp() + offset, structure)
    return tuple([stackData.__getattribute__(name) for name, type in stackData._fields_])