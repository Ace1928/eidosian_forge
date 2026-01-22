import functools
import unittest
from pecan import expose
from pecan import util
from pecan.compat import getargspec
def test_multiple_decorators(self):

    def dec(f):

        @functools.wraps(f)
        def wrapped(*a, **kw):
            return f(*a, **kw)
        return wrapped
    expected = getargspec(self.controller.index.__func__)
    actual = util.getargspec(dec(dec(dec(self.controller.index.__func__))))
    assert expected == actual
    expected = getargspec(self.controller.static_index)
    actual = util.getargspec(dec(dec(dec(self.controller.static_index))))
    assert expected == actual