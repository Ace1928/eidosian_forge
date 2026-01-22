import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@staticmethod
def dump_code_line(disassembly_line, bShowAddress=True, bShowDump=True, bLowercase=True, dwDumpWidth=None, dwCodeWidth=None, bits=None):
    """
        Dump a single line of code. To dump a block of code use L{dump_code}.

        @type  disassembly_line: tuple( int, int, str, str )
        @param disassembly_line: Single item of the list returned by
            L{Process.disassemble} or L{Thread.disassemble_around_pc}.

        @type  bShowAddress: bool
        @param bShowAddress: (Optional) If C{True} show the memory address.

        @type  bShowDump: bool
        @param bShowDump: (Optional) If C{True} show the hexadecimal dump.

        @type  bLowercase: bool
        @param bLowercase: (Optional) If C{True} convert the code to lowercase.

        @type  dwDumpWidth: int or None
        @param dwDumpWidth: (Optional) Width in characters of the hex dump.

        @type  dwCodeWidth: int or None
        @param dwCodeWidth: (Optional) Width in characters of the code.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
    if bits is None:
        address_size = HexDump.address_size
    else:
        address_size = bits / 4
    addr, size, code, dump = disassembly_line
    dump = dump.replace(' ', '')
    result = list()
    fmt = ''
    if bShowAddress:
        result.append(HexDump.address(addr, bits))
        fmt += '%%%ds:' % address_size
    if bShowDump:
        result.append(dump)
        if dwDumpWidth:
            fmt += ' %%-%ds' % dwDumpWidth
        else:
            fmt += ' %s'
    if bLowercase:
        code = code.lower()
    result.append(code)
    if dwCodeWidth:
        fmt += ' %%-%ds' % dwCodeWidth
    else:
        fmt += ' %s'
    return fmt % tuple(result)