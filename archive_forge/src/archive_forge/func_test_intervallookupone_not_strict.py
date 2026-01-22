from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq, eq_
from petl.util.vis import lookall
from petl.errors import DuplicateKeyError
from petl.transform.intervals import intervallookup, intervallookupone, \
def test_intervallookupone_not_strict():
    table = (('start', 'stop', 'value'), (1, 4, 'foo'), (3, 7, 'bar'), (4, 9, 'baz'))
    lkp = intervallookupone(table, 'start', 'stop', value='value', strict=False)
    actual = lkp.search(0, 1)
    expect = None
    eq_(expect, actual)
    actual = lkp.search(1, 2)
    expect = 'foo'
    eq_(expect, actual)
    actual = lkp.search(2, 4)
    expect = 'foo'
    eq_(expect, actual)
    actual = lkp.search(2, 5)
    expect = 'foo'
    eq_(expect, actual)
    actual = lkp.search(4, 5)
    expect = 'bar'
    eq_(expect, actual)
    actual = lkp.search(5, 7)
    expect = 'bar'
    eq_(expect, actual)
    actual = lkp.search(8, 9)
    expect = 'baz'
    eq_(expect, actual)
    actual = lkp.search(9, 14)
    expect = None
    eq_(expect, actual)
    actual = lkp.search(19, 140)
    expect = None
    eq_(expect, actual)
    actual = lkp.search(0)
    expect = None
    eq_(expect, actual)
    actual = lkp.search(1)
    expect = 'foo'
    eq_(expect, actual)
    actual = lkp.search(2)
    expect = 'foo'
    eq_(expect, actual)
    actual = lkp.search(4)
    expect = 'bar'
    eq_(expect, actual)
    actual = lkp.search(5)
    expect = 'bar'
    eq_(expect, actual)
    actual = lkp.search(8)
    expect = 'baz'
    eq_(expect, actual)