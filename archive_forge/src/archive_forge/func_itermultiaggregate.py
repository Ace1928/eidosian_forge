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
def itermultiaggregate(source, key, aggregation):
    aggregation = OrderedDict(aggregation.items())
    it = iter(source)
    hdr = next(it)
    it = itertools.chain([hdr], it)
    for outfld in aggregation:
        agg = aggregation[outfld]
        if callable(agg):
            aggregation[outfld] = (None, agg)
        elif isinstance(agg, string_types):
            aggregation[outfld] = (agg, list)
        elif len(agg) == 1 and isinstance(agg[0], string_types):
            aggregation[outfld] = (agg[0], list)
        elif len(agg) == 1 and callable(agg[0]):
            aggregation[outfld] = (None, agg[0])
        elif len(agg) == 2:
            pass
        else:
            raise ArgumentError('invalid aggregation: %r, %r' % (outfld, agg))
    if isinstance(key, (list, tuple)):
        outhdr = list(key)
    elif callable(key):
        outhdr = ['key']
    elif key is None:
        outhdr = []
    else:
        outhdr = [key]
    for outfld in aggregation:
        outhdr.append(outfld)
    yield tuple(outhdr)
    if key is None:
        grouped = rowgroupby(it, lambda x: None)
    else:
        grouped = rowgroupby(it, key)
    for k, rows in grouped:
        rows = list(rows)
        if isinstance(key, (list, tuple)):
            outrow = list(k)
        elif key is None:
            outrow = []
        else:
            outrow = [k]
        for outfld in aggregation:
            srcfld, aggfun = aggregation[outfld]
            if srcfld is None:
                aggval = aggfun(rows)
                outrow.append(aggval)
            elif isinstance(srcfld, (list, tuple)):
                idxs = [hdr.index(f) for f in srcfld]
                valgetter = operator.itemgetter(*idxs)
                vals = (valgetter(row) for row in rows)
                aggval = aggfun(vals)
                outrow.append(aggval)
            else:
                idx = hdr.index(srcfld)
                vals = (row[idx] for row in rows)
                aggval = aggfun(vals)
                outrow.append(aggval)
        yield tuple(outrow)