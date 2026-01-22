from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def fz_read_rbits(self, n):
    """
        Class-aware wrapper for `::fz_read_rbits()`.
        	Read the next n bits from a stream (assumed to
        	be packed least significant bit first).

        	stm: The stream to read from.

        	n: The number of bits to read, between 1 and 8*sizeof(int)
        	inclusive.

        	Returns (unsigned int)-1 for EOF, or the required number of bits.
        """
    return _mupdf.FzStream_fz_read_rbits(self, n)