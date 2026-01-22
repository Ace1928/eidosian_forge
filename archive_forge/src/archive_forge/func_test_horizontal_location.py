from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_horizontal_location():
    """Make sure we can move the cursor horizontally without changing rows."""
    t = TestTerminal(stream=StringIO(), force_styling=True)
    with t.location(x=5):
        pass
    eq_(t.stream.getvalue(), unicode_cap('sc') + unicode_parm('hpa', 5) + unicode_cap('rc'))