import zlib
from gitdb.util import byte_ord
import mmap
from itertools import islice
from functools import reduce
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_text
from gitdb.typ import (
from io import StringIO
def delta_list_apply(dcl, bbuf, write):
    """Apply the chain's changes and write the final result using the passed
    write function.
    :param bbuf: base buffer containing the base of all deltas contained in this
        list. It will only be used if the chunk in question does not have a base
        chain.
    :param write: function taking a string of bytes to write to the output"""
    for dc in dcl:
        delta_chunk_apply(dc, bbuf, write)