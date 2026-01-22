from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
def iteraddrownumbers(table, start, step, field):
    it = iter(table)
    try:
        hdr = next(it)
    except StopIteration:
        hdr = []
    outhdr = [field]
    outhdr.extend(hdr)
    yield tuple(outhdr)
    for row, n in izip(it, count(start, step)):
        outrow = [n]
        outrow.extend(row)
        yield tuple(outrow)