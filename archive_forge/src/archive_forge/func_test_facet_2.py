from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq, eq_
from petl.comparison import Comparable
from petl.transform.selects import select, selectin, selectcontains, \
def test_facet_2():
    table = (('foo', 'bar', 'baz'), ('aa', 4, 9.3), ('aa', 2, 88.2), ('bb', 1, 23.3), ('cc', 8, 42.0), ('dd', 7, 100.9), ('cc', 2))
    fct = facet(table, 'foo')
    assert set(fct.keys()) == {'aa', 'bb', 'cc', 'dd'}
    expect_fcta = (('foo', 'bar', 'baz'), ('aa', 4, 9.3), ('aa', 2, 88.2))
    ieq(fct['aa'], expect_fcta)
    ieq(fct['aa'], expect_fcta)
    expect_fctc = (('foo', 'bar', 'baz'), ('cc', 8, 42.0), ('cc', 2))
    ieq(fct['cc'], expect_fctc)
    ieq(fct['cc'], expect_fctc)