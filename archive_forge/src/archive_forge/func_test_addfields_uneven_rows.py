from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_addfields_uneven_rows():
    table = (('foo', 'bar'), ('M',), ('F', 34), ('-', 56, 'spong'))
    result = addfields(table, [('baz', 42), ('qux', 100), ('qux', 200)])
    expectation = (('foo', 'bar', 'baz', 'qux', 'qux'), ('M', None, 42, 100, 200), ('F', 34, 42, 100, 200), ('-', 56, 42, 100, 200))
    ieq(expectation, result)
    ieq(expectation, result)
    result = addfields(table, [('baz', 42), ('qux', 100, 0), ('qux', 200, 0)])
    expectation = (('qux', 'qux', 'foo', 'bar', 'baz'), (200, 100, 'M', None, 42), (200, 100, 'F', 34, 42), (200, 100, '-', 56, 42))
    ieq(expectation, result)
    ieq(expectation, result)