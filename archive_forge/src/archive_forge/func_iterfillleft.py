from __future__ import absolute_import, print_function, division
from petl.compat import next
from petl.util.base import Table, asindices
def iterfillleft(table, missing):
    it = iter(table)
    try:
        hdr = next(it)
    except StopIteration:
        return
    yield tuple(hdr)
    for row in it:
        outrow = list(reversed(row))
        for i, _ in enumerate(outrow):
            if i > 0 and outrow[i] == missing and (outrow[i - 1] != missing):
                outrow[i] = outrow[i - 1]
        yield tuple(reversed(outrow))