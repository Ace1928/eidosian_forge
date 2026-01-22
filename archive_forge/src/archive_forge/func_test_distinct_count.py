from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.dedup import duplicates, unique, conflicts, distinct, \
def test_distinct_count():
    table = (('foo', 'bar', 'baz'), (None, None, None), ('A', 1, 2), ('B', '2', '3.4'), ('B', '2', '3.4'), ('D', 4, 12.3))
    result = distinct(table, count='count')
    expect = (('foo', 'bar', 'baz', 'count'), (None, None, None, 1), ('A', 1, 2, 1), ('B', '2', '3.4', 2), ('D', 4, 12.3, 1))
    ieq(expect, result)