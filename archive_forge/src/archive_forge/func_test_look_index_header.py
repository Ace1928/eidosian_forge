from __future__ import absolute_import, print_function, division
import logging
from petl.test.helpers import eq_
import petl as etl
from petl.util.vis import look, see, lookstr
def test_look_index_header():
    table = (('foo', 'bar'), ('a', 1), ('b', 2))
    actual = repr(look(table, index_header=True))
    expect = "+-------+-------+\n| 0|foo | 1|bar |\n+=======+=======+\n| 'a'   |     1 |\n+-------+-------+\n| 'b'   |     2 |\n+-------+-------+\n"
    eq_(expect, actual)