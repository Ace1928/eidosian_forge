from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.test.helpers import ieq
from petl.util import strjoin
from petl.transform.reductions import rowreduce, aggregate, \
def test_aggregate_simple_key_is_None():
    table1 = (('foo', 'bar', 'baz'), ('a', 3, True), ('a', 7, False), ('b', 2, True), ('b', 2, False), ('b', 9, False), ('c', 4, True))
    table2 = aggregate(table1, None, len)
    expect2 = (('value',), (6,))
    ieq(expect2, table2)
    ieq(expect2, table2)
    table3 = aggregate(table1, None, sum, 'bar')
    expect3 = (('value',), (27,))
    ieq(expect3, table3)
    ieq(expect3, table3)
    table4 = aggregate(table1, key=None, aggregation=list, value=('bar', 'baz'))
    expect4 = (('value',), ([(3, True), (7, False), (2, True), (2, False), (9, False), (4, True)],))
    ieq(expect4, table4)
    ieq(expect4, table4)
    table5 = aggregate(table1, None, len, field='nrows')
    expect5 = (('nrows',), (6,))
    ieq(expect5, table5)
    ieq(expect5, table5)