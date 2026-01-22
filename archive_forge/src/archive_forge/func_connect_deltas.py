import zlib
from gitdb.util import byte_ord
import mmap
from itertools import islice
from functools import reduce
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_text
from gitdb.typ import (
from io import StringIO
def connect_deltas(dstreams):
    """
    Read the condensed delta chunk information from dstream and merge its information
        into a list of existing delta chunks

    :param dstreams: iterable of delta stream objects, the delta to be applied last
        comes first, then all its ancestors in order
    :return: DeltaChunkList, containing all operations to apply"""
    tdcl = None
    dcl = tdcl = TopdownDeltaChunkList()
    for dsi, ds in enumerate(dstreams):
        db = ds.read()
        delta_buf_size = ds.size
        i, base_size = msb_size(db)
        i, target_size = msb_size(db, i)
        tbw = 0
        while i < delta_buf_size:
            c = ord(db[i])
            i += 1
            if c & 128:
                cp_off, cp_size = (0, 0)
                if c & 1:
                    cp_off = ord(db[i])
                    i += 1
                if c & 2:
                    cp_off |= ord(db[i]) << 8
                    i += 1
                if c & 4:
                    cp_off |= ord(db[i]) << 16
                    i += 1
                if c & 8:
                    cp_off |= ord(db[i]) << 24
                    i += 1
                if c & 16:
                    cp_size = ord(db[i])
                    i += 1
                if c & 32:
                    cp_size |= ord(db[i]) << 8
                    i += 1
                if c & 64:
                    cp_size |= ord(db[i]) << 16
                    i += 1
                if not cp_size:
                    cp_size = 65536
                rbound = cp_off + cp_size
                if rbound < cp_size or rbound > base_size:
                    break
                dcl.append(DeltaChunk(tbw, cp_size, cp_off, None))
                tbw += cp_size
            elif c:
                dcl.append(DeltaChunk(tbw, c, 0, db[i:i + c]))
                i += c
                tbw += c
            else:
                raise ValueError('unexpected delta opcode 0')
        dcl.compress()
        if dsi > 0:
            if not tdcl.connect_with_next_base(dcl):
                break
        dcl = DeltaChunkList()
    return tdcl