from __future__ import absolute_import, print_function, division
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
from petl.util.counting import valuecount, valuecounter, valuecounts, \
def test_nrows():
    table = (('foo', 'bar'), ('a', 1), ('b',))
    actual = nrows(table)
    expect = 2
    eq_(expect, actual)