from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq, eq_
from petl.util.vis import lookall
from petl.errors import DuplicateKeyError
from petl.transform.intervals import intervallookup, intervallookupone, \
def test_intervallookup_include_stop():
    table = (('start', 'stop', 'value'), (1, 4, 'foo'), (3, 7, 'bar'), (4, 9, None))
    lkp = intervallookup(table, 'start', 'stop', value='value', include_stop=True)
    actual = lkp.search(0, 1)
    expect = ['foo']
    eq_(expect, actual)
    actual = lkp.search(1, 2)
    expect = ['foo']
    eq_(expect, actual)
    actual = lkp.search(2, 4)
    expect = ['foo', 'bar', None]
    eq_(expect, actual)
    actual = lkp.search(2, 5)
    expect = ['foo', 'bar', None]
    eq_(expect, actual)
    actual = lkp.search(9, 14)
    expect = [None]
    eq_(expect, actual)
    actual = lkp.search(19, 140)
    expect = []
    eq_(expect, actual)
    actual = lkp.search(1)
    expect = ['foo']
    eq_(expect, actual)
    actual = lkp.search(2)
    expect = ['foo']
    eq_(expect, actual)
    actual = lkp.search(4)
    expect = ['foo', 'bar', None]
    eq_(expect, actual)
    actual = lkp.search(5)
    expect = ['bar', None]
    eq_(expect, actual)