from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_cat_with_header():
    table1 = (('bar', 'foo'), ('A', 1), ('B', 2))
    table2 = (('bar', 'baz'), ('C', True), ('D', False))
    actual = cat(table1, header=['A', 'foo', 'B', 'bar', 'C'])
    expect = (('A', 'foo', 'B', 'bar', 'C'), (None, 1, None, 'A', None), (None, 2, None, 'B', None))
    ieq(expect, actual)
    ieq(expect, actual)
    actual = cat(table1, table2, header=['A', 'foo', 'B', 'bar', 'C'])
    expect = (('A', 'foo', 'B', 'bar', 'C'), (None, 1, None, 'A', None), (None, 2, None, 'B', None), (None, None, None, 'C', None), (None, None, None, 'D', None))
    ieq(expect, actual)
    ieq(expect, actual)