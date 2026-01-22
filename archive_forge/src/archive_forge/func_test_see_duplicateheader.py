from __future__ import absolute_import, print_function, division
import logging
from petl.test.helpers import eq_
import petl as etl
from petl.util.vis import look, see, lookstr
def test_see_duplicateheader():
    table = (('foo', 'bar', 'foo'), ('a', 1, 'a_prime'), ('b', 2, 'b_prime'))
    actual = repr(see(table))
    expect = "foo: 'a', 'b'\nbar: 1, 2\nfoo: 'a_prime', 'b_prime'\n"
    eq_(expect, actual)