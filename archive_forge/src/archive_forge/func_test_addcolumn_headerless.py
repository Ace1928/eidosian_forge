from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_addcolumn_headerless():
    """Adds a header row if none exists."""
    table1 = ()
    expect = (('foo',), ('A',), ('B',))
    actual = addcolumn(table1, 'foo', ['A', 'B'])
    ieq(expect, actual)
    ieq(expect, actual)