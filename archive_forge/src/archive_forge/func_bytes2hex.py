from contextlib import contextmanager
from .exceptions import ELFParseError, ELFError, DWARFError
from ..construct import ConstructError, ULInt8
import os
def bytes2hex(b, sep=''):
    if not sep:
        return b.hex()
    return sep.join(map('{:02x}'.format, b))