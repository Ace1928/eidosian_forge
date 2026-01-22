from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump, HexInput
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses
from winappdbg.module import Module, _ModuleContainer
from winappdbg.thread import Thread, _ThreadContainer
from winappdbg.window import Window
from winappdbg.search import Search, \
from winappdbg.disasm import Disassembler
import re
import os
import os.path
import ctypes
import struct
import warnings
import traceback
def take_memory_snapshot(self, minAddr=None, maxAddr=None):
    """
        Takes a snapshot of the memory contents of the process.

        It's best if the process is suspended (if alive) when taking the
        snapshot. Execution can be resumed afterwards.

        Example::
            # Print the memory contents of a process.
            process.suspend()
            try:
                snapshot = process.take_memory_snapshot()
                for mbi in snapshot:
                    print HexDump.hexblock(mbi.content, mbi.BaseAddress)
            finally:
                process.resume()

        You can also iterate the memory of a dead process, just as long as the
        last open handle to it hasn't been closed.

        @warning: If the target process has a very big memory footprint, the
            resulting snapshot will be equally big. This may result in a severe
            performance penalty.

        @see: L{generate_memory_snapshot}

        @type  minAddr: int
        @param minAddr: (Optional) Starting address in address range to query.

        @type  maxAddr: int
        @param maxAddr: (Optional) Ending address in address range to query.

        @rtype:  list( L{win32.MemoryBasicInformation} )
        @return: List of memory region information objects.
            Two extra properties are added to these objects:
             - C{filename}: Mapped filename, or C{None}.
             - C{content}: Memory contents, or C{None}.
        """
    return list(self.iter_memory_snapshot(minAddr, maxAddr))