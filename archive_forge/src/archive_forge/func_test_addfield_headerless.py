from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_addfield_headerless():
    """When adding a field to a headerless table, implicitly add a header."""
    table = ()
    expect = (('foo',),)
    actual = addfield(table, 'foo', 1)
    ieq(expect, actual)
    ieq(expect, actual)