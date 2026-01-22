from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq, eq_
from petl.util.vis import lookall
from petl.errors import DuplicateKeyError
from petl.transform.intervals import intervallookup, intervallookupone, \
def test_facetintervallookupone():
    table = (('type', 'start', 'stop', 'value'), ('apple', 1, 4, 'foo'), ('apple', 3, 7, 'bar'), ('orange', 4, 9, 'baz'))
    lkp = facetintervallookupone(table, key='type', start='start', stop='stop', value='value')
    actual = lkp['apple'].search(0, 1)
    expect = None
    eq_(expect, actual)
    actual = lkp['apple'].search(1, 2)
    expect = 'foo'
    eq_(expect, actual)
    try:
        lkp['apple'].search(2, 4)
    except DuplicateKeyError:
        pass
    else:
        assert False, 'expected error'
    try:
        lkp['apple'].search(2, 5)
    except DuplicateKeyError:
        pass
    else:
        assert False, 'expected error'
    actual = lkp['apple'].search(4, 5)
    expect = 'bar'
    eq_(expect, actual)
    actual = lkp['orange'].search(4, 5)
    expect = 'baz'
    eq_(expect, actual)
    actual = lkp['apple'].search(5, 7)
    expect = 'bar'
    eq_(expect, actual)
    actual = lkp['orange'].search(5, 7)
    expect = 'baz'
    eq_(expect, actual)
    actual = lkp['apple'].search(8, 9)
    expect = None
    eq_(expect, actual)
    actual = lkp['orange'].search(8, 9)
    expect = 'baz'
    eq_(expect, actual)
    actual = lkp['orange'].search(9, 14)
    expect = None
    eq_(expect, actual)
    actual = lkp['orange'].search(19, 140)
    expect = None
    eq_(expect, actual)
    actual = lkp['apple'].search(0)
    expect = None
    eq_(expect, actual)
    actual = lkp['apple'].search(1)
    expect = 'foo'
    eq_(expect, actual)
    actual = lkp['apple'].search(2)
    expect = 'foo'
    eq_(expect, actual)
    actual = lkp['apple'].search(4)
    expect = 'bar'
    eq_(expect, actual)
    actual = lkp['apple'].search(5)
    expect = 'bar'
    eq_(expect, actual)
    actual = lkp['orange'].search(5)
    expect = 'baz'
    eq_(expect, actual)
    actual = lkp['apple'].search(8)
    expect = None
    eq_(expect, actual)
    actual = lkp['orange'].search(8)
    expect = 'baz'
    eq_(expect, actual)