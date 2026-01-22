from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.util.base import Table
def iterunpackdict(table, field, keys, includeoriginal, samplesize, missing):
    it = iter(table)
    try:
        hdr = next(it)
    except StopIteration:
        hdr = []
    flds = list(map(text_type, hdr))
    fidx = flds.index(field)
    outhdr = list(flds)
    if not includeoriginal:
        del outhdr[fidx]
    if not keys:
        sample = list(itertools.islice(it, samplesize))
        keys = set()
        for row in sample:
            try:
                keys |= set(row[fidx].keys())
            except AttributeError:
                pass
        it = itertools.chain(sample, it)
        keys = sorted(keys)
    outhdr.extend(keys)
    yield tuple(outhdr)
    for row in it:
        outrow = list(row)
        if not includeoriginal:
            del outrow[fidx]
        for key in keys:
            try:
                outrow.append(row[fidx][key])
            except (IndexError, KeyError, TypeError):
                outrow.append(missing)
        yield tuple(outrow)