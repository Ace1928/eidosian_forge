from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_movefield():
    table1 = (('foo', 'bar', 'baz'), (1, 'A', True), (2, 'B', False))
    expect = (('bar', 'foo', 'baz'), ('A', 1, True), ('B', 2, False))
    actual = movefield(table1, 'bar', 0)
    ieq(expect, actual)
    ieq(expect, actual)
    actual = movefield(table1, 'foo', 1)
    ieq(expect, actual)
    ieq(expect, actual)