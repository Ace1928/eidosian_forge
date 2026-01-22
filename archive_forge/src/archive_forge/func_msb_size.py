import zlib
from gitdb.util import byte_ord
import mmap
from itertools import islice
from functools import reduce
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_text
from gitdb.typ import (
from io import StringIO
def msb_size(data, offset=0):
    """
    :return: tuple(read_bytes, size) read the msb size from the given random
        access data starting at the given byte offset"""
    size = 0
    i = 0
    l = len(data)
    hit_msb = False
    while i < l:
        c = data[i + offset]
        size |= (c & 127) << i * 7
        i += 1
        if not c & 128:
            hit_msb = True
            break
    if not hit_msb:
        raise AssertionError('Could not find terminating MSB byte in data stream')
    return (i + offset, size)