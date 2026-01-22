from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.compat import next, string_types, text_type
import petl.config as config
from petl.errors import ArgumentError
from petl.util.base import Table, expr, rowgroupby, Record
from petl.transform.sorts import sort
def iterrowmapmany(source, rowgenerator, header, failonerror):
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        return
    flds = list(map(text_type, hdr))
    yield tuple(header)
    it = (Record(row, flds) for row in it)
    for row in it:
        try:
            for outrow in rowgenerator(row):
                yield tuple(outrow)
        except Exception as e:
            if failonerror == 'inline':
                yield tuple([e])
            elif failonerror:
                raise e
            else:
                pass