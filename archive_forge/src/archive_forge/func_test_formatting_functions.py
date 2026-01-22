from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_formatting_functions():
    """Test crazy-ass formatting wrappers, both simple and compound."""
    t = TestTerminal()
    eq_(t.bold(u'hi'), t.bold + u'hi' + t.normal)
    eq_(t.green('hi'), t.green + u'hi' + t.normal)
    eq_(t.bold_green(u'boö'), t.bold + t.green + u'boö' + t.normal)
    eq_(t.bold_underline_green_on_red('boo'), t.bold + t.underline + t.green + t.on_red + u'boo' + t.normal)
    eq_(t.on_bright_red_bold_bright_green_underline('meh'), t.on_bright_red + t.bold + t.bright_green + t.underline + u'meh' + t.normal)