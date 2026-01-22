from __future__ import absolute_import, print_function, division
from datetime import datetime
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.reshape import melt, recast, transpose, pivot, flatten, \
from petl.transform.regex import split, capture
def test_flatten_empty():
    table1 = (('foo', 'bar', 'baz'),)
    expect1 = []
    actual1 = flatten(table1)
    ieq(expect1, actual1)