from __future__ import absolute_import, print_function, division
import logging
from petl.test.helpers import eq_
import petl as etl
from petl.util.vis import look, see, lookstr
def test_see():
    table = (('foo', 'bar'), ('a', 1), ('b', 2))
    actual = repr(see(table))
    expect = "foo: 'a', 'b'\nbar: 1, 2\n"
    eq_(expect, actual)