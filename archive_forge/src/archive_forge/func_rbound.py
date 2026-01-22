import zlib
from gitdb.util import byte_ord
import mmap
from itertools import islice
from functools import reduce
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_text
from gitdb.typ import (
from io import StringIO
def rbound(self):
    """:return: rightmost extend in bytes, absolute"""
    if len(self) == 0:
        return 0
    return self[-1].rbound()