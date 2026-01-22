from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def disassemble_current(self):
    """
        Disassemble the instruction at the program counter of the given thread.

        @rtype:  tuple( long, int, str, str )
        @return: The tuple represents an assembly instruction
            and contains:
             - Memory address of instruction.
             - Size of instruction in bytes.
             - Disassembly line of instruction.
             - Hexadecimal dump of instruction.
        """
    return self.disassemble_instruction(self.get_pc())