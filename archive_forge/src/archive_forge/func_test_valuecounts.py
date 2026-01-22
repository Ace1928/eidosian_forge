from __future__ import absolute_import, print_function, division
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
from petl.util.counting import valuecount, valuecounter, valuecounts, \
def test_valuecounts():
    table = (('foo', 'bar'), ('a', 1), ('b', 2), ('b', 7))
    actual = valuecounts(table, 'foo')
    expect = (('foo', 'count', 'frequency'), ('b', 2, 2.0 / 3), ('a', 1, 1.0 / 3))
    ieq(expect, actual)
    ieq(expect, actual)