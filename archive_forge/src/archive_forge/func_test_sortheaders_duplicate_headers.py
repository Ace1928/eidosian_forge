from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import ieq
from petl.errors import FieldSelectionError
from petl.util import fieldnames
from petl.transform.headers import setheader, extendheader, pushheader, skip, \
def test_sortheaders_duplicate_headers():
    """ Failing test case provided in sortheader()
    with duplicate column names overlays values #392 """
    table1 = (('id', 'foo', 'foo', 'foo'), ('a', 1, 2, 3), ('b', 4, 5, 6))
    expect = (('foo', 'foo', 'foo', 'id'), (1, 2, 3, 'a'), (4, 5, 6, 'b'))
    actual = sortheader(table1)
    ieq(expect, actual)