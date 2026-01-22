from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.test.helpers import ieq
from petl.util import strjoin
from petl.transform.reductions import rowreduce, aggregate, \
def test_aggregate_more():
    table1 = (('foo', 'bar'), ('aa', 3), ('aa', 7), ('bb', 2), ('bb', 1), ('bb', 9), ('cc', 4), ('dd', 3))
    aggregators = OrderedDict()
    aggregators['minbar'] = ('bar', min)
    aggregators['maxbar'] = ('bar', max)
    aggregators['sumbar'] = ('bar', sum)
    aggregators['listbar'] = 'bar'
    aggregators['bars'] = ('bar', strjoin(', '))
    table2 = aggregate(table1, 'foo', aggregators)
    expect2 = (('foo', 'minbar', 'maxbar', 'sumbar', 'listbar', 'bars'), ('aa', 3, 7, 10, [3, 7], '3, 7'), ('bb', 1, 9, 12, [2, 1, 9], '2, 1, 9'), ('cc', 4, 4, 4, [4], '4'), ('dd', 3, 3, 3, [3], '3'))
    ieq(expect2, table2)
    ieq(expect2, table2)
    table3 = aggregate(table1, 'foo')
    table3['minbar'] = ('bar', min)
    table3['maxbar'] = ('bar', max)
    table3['sumbar'] = ('bar', sum)
    table3['listbar'] = 'bar'
    table3['bars'] = ('bar', strjoin(', '))
    ieq(expect2, table3)