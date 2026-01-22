from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_cut_empty():
    table = (('foo', 'bar'),)
    expect = (('bar',),)
    actual = cut(table, 'bar')
    ieq(expect, actual)