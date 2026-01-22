import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
def bits_to_int(bits, dialect=None):
    if dialect is None:
        dialect = DEFAULT_EUI64_DIALECT
    return _bits_to_int(bits, width, dialect.word_sep)