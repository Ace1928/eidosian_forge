from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.compat import next, string_types, text_type
import petl.config as config
from petl.errors import ArgumentError
from petl.util.base import Table, expr, rowgroupby, Record
from petl.transform.sorts import sort
def iterfieldmap(source, mappings, failonerror, errorvalue):
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        return
    flds = list(map(text_type, hdr))
    outhdr = mappings.keys()
    yield tuple(outhdr)
    mapfuns = dict()
    for outfld, m in mappings.items():
        if m in hdr:
            mapfuns[outfld] = operator.itemgetter(m)
        elif isinstance(m, int) and m < len(hdr):
            mapfuns[outfld] = operator.itemgetter(m)
        elif isinstance(m, string_types):
            mapfuns[outfld] = expr(m)
        elif callable(m):
            mapfuns[outfld] = m
        elif isinstance(m, (tuple, list)) and len(m) == 2:
            srcfld = m[0]
            fm = m[1]
            if callable(fm):
                mapfuns[outfld] = composefun(fm, srcfld)
            elif isinstance(fm, dict):
                mapfuns[outfld] = composedict(fm, srcfld)
            else:
                raise ArgumentError('expected callable or dict')
        else:
            raise ArgumentError('invalid mapping %r: %r' % (outfld, m))
    it = (Record(row, flds) for row in it)
    for row in it:
        outrow = list()
        for outfld in outhdr:
            try:
                val = mapfuns[outfld](row)
            except Exception as e:
                if failonerror == 'inline':
                    val = e
                elif failonerror:
                    raise e
                else:
                    val = errorvalue
            outrow.append(val)
        yield tuple(outrow)