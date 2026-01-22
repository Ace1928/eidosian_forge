from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq, eq_
from petl.util.vis import lookall
from petl.errors import DuplicateKeyError
from petl.transform.intervals import intervallookup, intervallookupone, \
def test_intervalantijoin():
    left = (('begin', 'end', 'quux'), (1, 2, 'a'), (2, 4, 'b'), (2, 5, 'c'), (9, 14, 'd'), (9, 140, 'e'), (1, 1, 'f'), (2, 2, 'g'), (4, 4, 'h'), (5, 5, 'i'), (1, 8, 'j'))
    right = (('start', 'stop', 'value'), (1, 4, 'foo'), (3, 7, 'bar'), (4, 9, 'baz'))
    actual = intervalantijoin(left, right, lstart='begin', lstop='end', rstart='start', rstop='stop')
    expect = (('begin', 'end', 'quux'), (9, 14, 'd'), (9, 140, 'e'), (1, 1, 'f'), (2, 2, 'g'), (4, 4, 'h'), (5, 5, 'i'))
    debug(lookall(actual))
    ieq(expect, actual)
    ieq(expect, actual)