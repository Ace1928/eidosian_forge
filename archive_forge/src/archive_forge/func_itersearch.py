from __future__ import absolute_import, print_function, division
import re
import operator
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.util.base import Table, asindices
from petl.transform.basics import TransformError
from petl.transform.conversions import convert
def itersearch(table, pattern, field, flags, complement):
    prog = re.compile(pattern, flags)
    it = iter(table)
    try:
        hdr = next(it)
    except StopIteration:
        return
    flds = list(map(text_type, hdr))
    yield tuple(hdr)
    if field is None:
        test = lambda r: any((prog.search(text_type(v)) for v in r))
    else:
        indices = asindices(hdr, field)
        if len(indices) == 1:
            index = indices[0]
            test = lambda r: prog.search(text_type(r[index]))
        else:
            getvals = operator.itemgetter(*indices)
            test = lambda r: any((prog.search(text_type(v)) for v in getvals(r)))
    if not complement:
        for row in it:
            if test(row):
                yield tuple(row)
    else:
        for row in it:
            if not test(row):
                yield tuple(row)