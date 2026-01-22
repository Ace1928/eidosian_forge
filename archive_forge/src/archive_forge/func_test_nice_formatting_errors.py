from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_nice_formatting_errors():
    """Make sure you get nice hints if you misspell a formatting wrapper."""
    t = TestTerminal()
    try:
        t.bold_misspelled('hey')
    except TypeError as e:
        assert 'probably misspelled' in e.args[0]
    try:
        t.bold_misspelled(u'hey')
    except TypeError as e:
        assert 'probably misspelled' in e.args[0]
    try:
        t.bold_misspelled(None)
    except TypeError as e:
        assert 'probably misspelled' not in e.args[0]
    try:
        t.bold_misspelled('a', 'b')
    except TypeError as e:
        assert 'probably misspelled' not in e.args[0]