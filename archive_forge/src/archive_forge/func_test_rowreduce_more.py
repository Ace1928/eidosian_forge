from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.test.helpers import ieq
from petl.util import strjoin
from petl.transform.reductions import rowreduce, aggregate, \
def test_rowreduce_more():
    table1 = (('foo', 'bar'), ('aa', 3), ('aa', 7), ('bb', 2), ('bb', 1), ('bb', 9), ('cc', 4))

    def sumbar(key, records):
        return [key, sum((rec['bar'] for rec in records))]
    table2 = rowreduce(table1, key='foo', reducer=sumbar, header=['foo', 'barsum'])
    expect2 = (('foo', 'barsum'), ('aa', 10), ('bb', 12), ('cc', 4))
    ieq(expect2, table2)