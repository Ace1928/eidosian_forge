from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_null_callable_numeric_colors():
    """``color(n)`` should be a no-op on null terminals."""
    t = TestTerminal(stream=StringIO())
    eq_(t.color(5)('smoo'), 'smoo')
    eq_(t.on_color(6)('smoo'), 'smoo')