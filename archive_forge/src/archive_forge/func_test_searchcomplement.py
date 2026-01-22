from __future__ import absolute_import, print_function, division
import pytest
from petl.compat import next
from petl.errors import ArgumentError
from petl.test.helpers import ieq, eq_
from petl.transform.regex import capture, split, search, searchcomplement, splitdown
from petl.transform.basics import TransformError
def test_searchcomplement():
    table1 = (('foo', 'bar', 'baz'), ('orange', 12, 'oranges are nice fruit'), ('mango', 42, 'I like them'), ('banana', 74, 'lovely too'), ('cucumber', 41, 'better than mango'))
    table2 = searchcomplement(table1, '.g.')
    expect2 = (('foo', 'bar', 'baz'), ('banana', 74, 'lovely too'))
    ieq(expect2, table2)
    ieq(expect2, table2)
    table3 = searchcomplement(table1, 'foo', '.g.')
    expect3 = (('foo', 'bar', 'baz'), ('banana', 74, 'lovely too'), ('cucumber', 41, 'better than mango'))
    ieq(expect3, table3)
    ieq(expect3, table3)
    table2 = search(table1, '.g.', complement=True)
    expect2 = (('foo', 'bar', 'baz'), ('banana', 74, 'lovely too'))
    ieq(expect2, table2)
    ieq(expect2, table2)
    table3 = search(table1, 'foo', '.g.', complement=True)
    expect3 = (('foo', 'bar', 'baz'), ('banana', 74, 'lovely too'), ('cucumber', 41, 'better than mango'))
    ieq(expect3, table3)
    ieq(expect3, table3)