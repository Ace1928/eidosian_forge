from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_addfield_uneven_rows():
    table = (('foo', 'bar'), ('M',), ('F', 34), ('-', 56, 'spong'))
    result = addfield(table, 'baz', 42)
    expectation = (('foo', 'bar', 'baz'), ('M', None, 42), ('F', 34, 42), ('-', 56, 42))
    ieq(expectation, result)
    ieq(expectation, result)