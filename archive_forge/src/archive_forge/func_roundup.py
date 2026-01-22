from contextlib import contextmanager
from .exceptions import ELFParseError, ELFError, DWARFError
from ..construct import ConstructError, ULInt8
import os
def roundup(num, bits):
    """ Round up a number to nearest multiple of 2^bits. The result is a number
        where the least significant bits passed in bits are 0.
    """
    return (num - 1 | (1 << bits) - 1) + 1