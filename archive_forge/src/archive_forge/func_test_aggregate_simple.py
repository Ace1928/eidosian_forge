from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.test.helpers import ieq
from petl.util import strjoin
from petl.transform.reductions import rowreduce, aggregate, \
def test_aggregate_simple():
    table1 = (('foo', 'bar', 'baz'), ('a', 3, True), ('a', 7, False), ('b', 2, True), ('b', 2, False), ('b', 9, False), ('c', 4, True))
    table2 = aggregate(table1, 'foo', len)
    expect2 = (('foo', 'value'), ('a', 2), ('b', 3), ('c', 1))
    ieq(expect2, table2)
    ieq(expect2, table2)
    table3 = aggregate(table1, 'foo', sum, 'bar')
    expect3 = (('foo', 'value'), ('a', 10), ('b', 13), ('c', 4))
    ieq(expect3, table3)
    ieq(expect3, table3)
    table4 = aggregate(table1, key=('foo', 'bar'), aggregation=list, value=('bar', 'baz'))
    expect4 = (('foo', 'bar', 'value'), ('a', 3, [(3, True)]), ('a', 7, [(7, False)]), ('b', 2, [(2, True), (2, False)]), ('b', 9, [(9, False)]), ('c', 4, [(4, True)]))
    ieq(expect4, table4)
    ieq(expect4, table4)
    table5 = aggregate(table1, 'foo', len, field='nrows')
    expect5 = (('foo', 'nrows'), ('a', 2), ('b', 3), ('c', 1))
    ieq(expect5, table5)
    ieq(expect5, table5)