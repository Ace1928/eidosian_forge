import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@classmethod
def integer_list_file(cls, filename, values, bits=None):
    """
        Write a list of integers to a file.
        If a file of the same name exists, it's contents are replaced.

        See L{HexInput.integer_list_file} for a description of the file format.

        @type  filename: str
        @param filename: Name of the file to write.

        @type  values: list( int )
        @param values: List of integers to write to the file.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexOutput.integer_size}
        """
    fd = open(filename, 'w')
    for integer in values:
        (print >> fd, cls.integer(integer, bits))
    fd.close()