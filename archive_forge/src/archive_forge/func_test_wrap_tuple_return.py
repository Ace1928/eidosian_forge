from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import csv
from petl.compat import PY2
import petl as etl
from petl.test.helpers import ieq, eq_
def test_wrap_tuple_return():
    tablea = etl.wrap((('foo', 'bar'), ('A', 1), ('C', 7)))
    tableb = etl.wrap((('foo', 'bar'), ('B', 5), ('C', 7)))
    added, removed = tablea.diff(tableb)
    eq_(('foo', 'bar'), added.header())
    eq_(('foo', 'bar'), removed.header())
    ieq(etl.data(added), added.data())
    ieq(etl.data(removed), removed.data())