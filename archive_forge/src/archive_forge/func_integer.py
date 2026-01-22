import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@classmethod
def integer(cls, integer, bits=None):
    """
        @type  integer: int
        @param integer: Integer.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.integer_size}

        @rtype:  str
        @return: Text output.
        """
    if bits is None:
        integer_size = cls.integer_size
    else:
        integer_size = bits / 4
    return '%%.%dX' % integer_size % integer