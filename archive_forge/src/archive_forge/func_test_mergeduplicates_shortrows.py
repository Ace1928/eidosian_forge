from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.test.helpers import ieq
from petl.util import strjoin
from petl.transform.reductions import rowreduce, aggregate, \
def test_mergeduplicates_shortrows():
    table = [['foo', 'bar', 'baz'], ['a', 1, True], ['b', 2, True], ['b', 3]]
    actual = mergeduplicates(table, 'foo')
    expect = [('foo', 'bar', 'baz'), ('a', 1, True), ('b', Conflict([2, 3]), True)]
    ieq(expect, actual)