from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_number_of_colors_with_tty():
    """``number_of_colors`` should work."""
    t = TestTerminal()
    eq_(t.number_of_colors, 256)