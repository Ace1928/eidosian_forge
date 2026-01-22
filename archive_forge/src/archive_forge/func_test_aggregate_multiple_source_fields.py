from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.test.helpers import ieq
from petl.util import strjoin
from petl.transform.reductions import rowreduce, aggregate, \
def test_aggregate_multiple_source_fields():
    table = (('foo', 'bar', 'baz'), ('a', 3, True), ('a', 7, False), ('b', 2, True), ('b', 2, False), ('b', 9, False), ('c', 4, True))
    expect = (('foo', 'bar', 'value'), ('a', 3, [(3, True)]), ('a', 7, [(7, False)]), ('b', 2, [(2, True), (2, False)]), ('b', 9, [(9, False)]), ('c', 4, [(4, True)]))
    actual = aggregate(table, ('foo', 'bar'), list, ('bar', 'baz'))
    ieq(expect, actual)
    ieq(expect, actual)
    actual = aggregate(table, key=('foo', 'bar'), aggregation=list, value=('bar', 'baz'))
    ieq(expect, actual)
    ieq(expect, actual)
    actual = aggregate(table, key=('foo', 'bar'))
    actual['value'] = (('bar', 'baz'), list)
    ieq(expect, actual)
    ieq(expect, actual)