from __future__ import absolute_import, print_function, division
import logging
from petl.test.helpers import eq_
import petl as etl
from petl.util.vis import look, see, lookstr
def test_look_headerless():
    table = []
    actual = repr(look(table))
    expect = ''
    eq_(expect, actual)