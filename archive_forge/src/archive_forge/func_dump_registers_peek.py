import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@staticmethod
def dump_registers_peek(registers, data, separator=' ', width=16):
    """
        Dump data pointed to by the given registers, if any.

        @type  registers: dict( str S{->} int )
        @param registers: Dictionary mapping register names to their values.
            This value is returned by L{Thread.get_context}.

        @type  data: dict( str S{->} str )
        @param data: Dictionary mapping register names to the data they point to.
            This value is returned by L{Thread.peek_pointers_in_registers}.

        @rtype:  str
        @return: Text suitable for logging.
        """
    if None in (registers, data):
        return ''
    names = compat.keys(data)
    names.sort()
    result = ''
    for reg_name in names:
        tag = reg_name.lower()
        dumped = HexDump.hexline(data[reg_name], separator, width)
        result += '%s -> %s\n' % (tag, dumped)
    return result