from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_parametrization():
    """Test parametrizing a capability."""
    eq_(TestTerminal().cup(3, 4), unicode_parm('cup', 3, 4))