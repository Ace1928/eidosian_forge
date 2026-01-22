from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq, eq_
from petl.comparison import Comparable
from petl.transform.selects import select, selectin, selectcontains, \
def test_facet():
    table = (('foo', 'bar', 'baz'), ('a', 4, 9.3), ('a', 2, 88.2), ('b', 1, 23.3), ('c', 8, 42.0), ('d', 7, 100.9), ('c', 2))
    fct = facet(table, 'foo')
    assert set(fct.keys()) == {'a', 'b', 'c', 'd'}
    expect_fcta = (('foo', 'bar', 'baz'), ('a', 4, 9.3), ('a', 2, 88.2))
    ieq(fct['a'], expect_fcta)
    ieq(fct['a'], expect_fcta)
    expect_fctc = (('foo', 'bar', 'baz'), ('c', 8, 42.0), ('c', 2))
    ieq(fct['c'], expect_fctc)
    ieq(fct['c'], expect_fctc)