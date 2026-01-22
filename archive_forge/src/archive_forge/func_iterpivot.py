from __future__ import absolute_import, print_function, division
import itertools
import collections
import operator
from petl.compat import next, text_type
from petl.comparison import comparable_itemgetter
from petl.util.base import Table, rowgetter, values, itervalues, \
from petl.transform.sorts import sort
def iterpivot(source, f1, f2, f3, aggfun, missing):
    f2vals = set(itervalues(source, f2))
    f2vals = list(f2vals)
    f2vals.sort()
    outhdr = [f1]
    outhdr.extend(f2vals)
    yield tuple(outhdr)
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        hdr = []
    flds = list(map(text_type, hdr))
    f1i = flds.index(f1)
    f2i = flds.index(f2)
    f3i = flds.index(f3)
    for v1, v1rows in itertools.groupby(it, key=operator.itemgetter(f1i)):
        outrow = [v1] + [missing] * len(f2vals)
        for v2, v12rows in itertools.groupby(v1rows, key=operator.itemgetter(f2i)):
            aggval = aggfun([row[f3i] for row in v12rows])
            outrow[1 + f2vals.index(v2)] = aggval
        yield tuple(outrow)