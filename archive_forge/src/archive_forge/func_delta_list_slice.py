import zlib
from gitdb.util import byte_ord
import mmap
from itertools import islice
from functools import reduce
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_text
from gitdb.typ import (
from io import StringIO
def delta_list_slice(dcl, absofs, size, ndcl):
    """:return: Subsection of this  list at the given absolute  offset, with the given
        size in bytes.
    :return: None"""
    cdi = _closest_index(dcl, absofs)
    cd = dcl[cdi]
    slen = len(dcl)
    lappend = ndcl.append
    if cd.to != absofs:
        tcd = DeltaChunk(cd.to, cd.ts, cd.so, cd.data)
        _move_delta_lbound(tcd, absofs - cd.to)
        tcd.ts = min(tcd.ts, size)
        lappend(tcd)
        size -= tcd.ts
        cdi += 1
    while cdi < slen and size:
        cd = dcl[cdi]
        if cd.ts <= size:
            lappend(DeltaChunk(cd.to, cd.ts, cd.so, cd.data))
            size -= cd.ts
        else:
            tcd = DeltaChunk(cd.to, cd.ts, cd.so, cd.data)
            tcd.ts = size
            lappend(tcd)
            size -= tcd.ts
            break
        cdi += 1