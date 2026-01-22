from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq, eq_
from petl.util.vis import lookall
from petl.errors import DuplicateKeyError
from petl.transform.intervals import intervallookup, intervallookupone, \
def test_facetintervallookup_compound():
    table = (('type', 'variety', 'start', 'stop', 'value'), ('apple', 'cox', 1, 4, 'foo'), ('apple', 'fuji', 3, 7, 'bar'), ('orange', 'mandarin', 4, 9, 'baz'))
    lkp = facetintervallookup(table, key=('type', 'variety'), start='start', stop='stop')
    actual = lkp['apple', 'cox'].search(1, 2)
    expect = [('apple', 'cox', 1, 4, 'foo')]
    eq_(expect, actual)
    actual = lkp['apple', 'cox'].search(2, 4)
    expect = [('apple', 'cox', 1, 4, 'foo')]
    eq_(expect, actual)