from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_rowslice():
    table = (('foo', 'bar', 'baz'), ('A', 1, 2), ('B', '2', '3.4'), (u'B', u'3', u'7.8', True), ('D', 'xyz', 9.0), ('E', None))
    result = rowslice(table, 2)
    expectation = (('foo', 'bar', 'baz'), ('A', 1, 2), ('B', '2', '3.4'))
    ieq(expectation, result)
    result = rowslice(table, 1, 2)
    expectation = (('foo', 'bar', 'baz'), ('B', '2', '3.4'))
    ieq(expectation, result)
    result = rowslice(table, 1, 5, 2)
    expectation = (('foo', 'bar', 'baz'), ('B', '2', '3.4'), ('D', 'xyz', 9.0))
    ieq(expectation, result)