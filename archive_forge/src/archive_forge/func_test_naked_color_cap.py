from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_naked_color_cap():
    """``term.color`` should return a stringlike capability."""
    t = TestTerminal()
    eq_(t.color + '', t.setaf + '')