from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
def iterrowslice(source, sliceargs):
    it = iter(source)
    try:
        yield tuple(next(it))
    except StopIteration:
        return
    for row in islice(it, *sliceargs):
        yield tuple(row)