from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import csv
from petl.compat import PY2
import petl as etl
from petl.test.helpers import ieq, eq_
def test_container():
    table = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2))
    actual = etl.wrap(table)[0]
    expect = ('foo', 'bar')
    eq_(expect, actual)
    actual = etl.wrap(table)['bar']
    expect = (1, 2, 2)
    ieq(expect, actual)
    actual = len(etl.wrap(table))
    expect = 4
    eq_(expect, actual)