from __future__ import absolute_import, print_function, division
import itertools
import operator
from collections import OrderedDict
from petl.compat import next, string_types, reduce, text_type
from petl.errors import ArgumentError
from petl.util.base import Table, iterpeek, rowgroupby
from petl.util.base import values
from petl.util.counting import nrows
from petl.transform.sorts import sort, mergesort
from petl.transform.basics import cut
from petl.transform.dedup import distinct
def itermergeduplicates(table, key, missing):
    it = iter(table)
    hdr, it = iterpeek(it)
    flds = list(map(text_type, hdr))
    if isinstance(key, string_types):
        outhdr = [key]
        keyflds = {key}
    else:
        outhdr = list(key)
        keyflds = set(key)
    valflds = [f for f in flds if f not in keyflds]
    valfldidxs = [flds.index(f) for f in valflds]
    outhdr.extend(valflds)
    yield tuple(outhdr)
    for k, grp in rowgroupby(it, key):
        grp = list(grp)
        if isinstance(key, string_types):
            outrow = [k]
        else:
            outrow = list(k)
        mergedvals = [set((row[i] for row in grp if len(row) > i and row[i] != missing)) for i in valfldidxs]
        normedvals = [vals.pop() if len(vals) == 1 else missing if len(vals) == 0 else Conflict(vals) for vals in mergedvals]
        outrow.extend(normedvals)
        yield tuple(outrow)