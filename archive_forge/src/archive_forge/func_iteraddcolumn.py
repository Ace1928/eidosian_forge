from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
def iteraddcolumn(table, field, col, index, missing):
    it = iter(table)
    try:
        hdr = next(it)
    except StopIteration:
        hdr = []
    if index is None:
        index = len(hdr)
    outhdr = list(hdr)
    outhdr.insert(index, field)
    yield tuple(outhdr)
    for row, val in izip_longest(it, col, fillvalue=missing):
        if row == missing:
            row = [missing] * len(hdr)
        outrow = list(row)
        outrow.insert(index, val)
        yield tuple(outrow)