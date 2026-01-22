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
def itersimpleaggregate(table, key, aggregation, value, field):
    if aggregation == len and key is not None:
        aggregation = lambda g: sum((1 for _ in g))
    if isinstance(key, (list, tuple)) and len(key) == 1:
        key = key[0]
    if isinstance(key, (list, tuple)):
        outhdr = tuple(key) + (field,)
    elif callable(key):
        outhdr = ('key', field)
    elif key is None:
        outhdr = (field,)
    else:
        outhdr = (key, field)
    yield outhdr
    if isinstance(key, (list, tuple)):
        for k, grp in rowgroupby(table, key, value):
            yield (tuple(k) + (aggregation(grp),))
    elif key is None:
        if aggregation == len:
            yield (nrows(table),)
        else:
            yield (aggregation(values(table, value)),)
    else:
        for k, grp in rowgroupby(table, key, value):
            yield (k, aggregation(grp))