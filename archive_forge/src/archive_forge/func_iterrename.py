from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import next, text_type
from petl.errors import FieldSelectionError
from petl.util.base import Table, asindices, rowgetter
def iterrename(source, spec, strict):
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        hdr = []
    flds = list(map(text_type, hdr))
    if strict:
        for x in spec:
            if isinstance(x, int):
                if x < 0 or x >= len(hdr):
                    raise FieldSelectionError(x)
            elif x not in flds:
                raise FieldSelectionError(x)
    outhdr = [spec[i] if i in spec else spec[f] if f in spec else f for i, f in enumerate(flds)]
    yield tuple(outhdr)
    for row in it:
        yield tuple(row)