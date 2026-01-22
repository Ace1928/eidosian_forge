from __future__ import absolute_import, print_function, division
import re
import operator
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.util.base import Table, asindices
from petl.transform.basics import TransformError
from petl.transform.conversions import convert
def itersplitdown(table, field, pattern, maxsplit, flags):
    prog = re.compile(pattern, flags)
    it = iter(table)
    try:
        hdr = next(it)
    except StopIteration:
        return
    flds = list(map(text_type, hdr))
    if isinstance(field, int) and field < len(hdr):
        field_index = field
        field = hdr[field_index]
    elif field in flds:
        field_index = flds.index(field)
    else:
        raise ArgumentError('field invalid: must be either field name or index')
    yield tuple(hdr)
    for row in it:
        value = row[field_index]
        for v in prog.split(value, maxsplit):
            yield tuple((v if i == field_index else row[i] for i in range(len(hdr))))