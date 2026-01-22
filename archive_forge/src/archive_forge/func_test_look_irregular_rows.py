from __future__ import absolute_import, print_function, division
import logging
from petl.test.helpers import eq_
import petl as etl
from petl.util.vis import look, see, lookstr
def test_look_irregular_rows():
    table = (('foo', 'bar'), ('a',), ('b', 2, True))
    actual = repr(look(table))
    expect = "+-----+-----+------+\n| foo | bar |      |\n+=====+=====+======+\n| 'a' |     |      |\n+-----+-----+------+\n| 'b' |   2 | True |\n+-----+-----+------+\n"
    eq_(expect, actual)