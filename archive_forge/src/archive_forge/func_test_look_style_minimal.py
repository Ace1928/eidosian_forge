from __future__ import absolute_import, print_function, division
import logging
from petl.test.helpers import eq_
import petl as etl
from petl.util.vis import look, see, lookstr
def test_look_style_minimal():
    table = (('foo', 'bar'), ('a', 1), ('b', 2))
    actual = repr(look(table, style='minimal'))
    expect = "foo  bar\n'a'    1\n'b'    2\n"
    eq_(expect, actual)
    etl.config.look_style = 'minimal'
    actual = repr(look(table))
    eq_(expect, actual)
    etl.config.look_style = 'grid'