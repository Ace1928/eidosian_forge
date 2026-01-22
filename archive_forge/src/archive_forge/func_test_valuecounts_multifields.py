from __future__ import absolute_import, print_function, division
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
from petl.util.counting import valuecount, valuecounter, valuecounts, \
def test_valuecounts_multifields():
    table = (('foo', 'bar', 'baz'), ('a', True, 0.12), ('a', True, 0.17), ('b', False, 0.34), ('b', False, 0.44), ('b',), ('b', False, 0.56))
    actual = valuecounts(table, 'foo', 'bar')
    expect = (('foo', 'bar', 'count', 'frequency'), ('b', False, 3, 3.0 / 6), ('a', True, 2, 2.0 / 6), ('b', None, 1, 1.0 / 6))
    ieq(expect, actual)
    ieq(expect, actual)