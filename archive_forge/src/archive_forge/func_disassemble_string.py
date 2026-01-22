from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def disassemble_string(self, lpAddress, code):
    """
        Disassemble instructions from a block of binary code.

        @type  lpAddress: int
        @param lpAddress: Memory address where the code was read from.

        @type  code: str
        @param code: Binary code to disassemble.

        @rtype:  list of tuple( long, int, str, str )
        @return: List of tuples. Each tuple represents an assembly instruction
            and contains:
             - Memory address of instruction.
             - Size of instruction in bytes.
             - Disassembly line of instruction.
             - Hexadecimal dump of instruction.
        """
    aProcess = self.get_process()
    return aProcess.disassemble_string(lpAddress, code)