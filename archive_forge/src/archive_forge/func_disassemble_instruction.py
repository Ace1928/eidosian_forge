from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def disassemble_instruction(self, lpAddress):
    """
        Disassemble the instruction at the given memory address.

        @type  lpAddress: int
        @param lpAddress: Memory address where to read the code from.

        @rtype:  tuple( long, int, str, str )
        @return: The tuple represents an assembly instruction
            and contains:
             - Memory address of instruction.
             - Size of instruction in bytes.
             - Disassembly line of instruction.
             - Hexadecimal dump of instruction.
        """
    aProcess = self.get_process()
    return aProcess.disassemble(lpAddress, 15)[0]