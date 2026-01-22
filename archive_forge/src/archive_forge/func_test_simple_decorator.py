import functools
import unittest
from pecan import expose
from pecan import util
from pecan.compat import getargspec
def test_simple_decorator(self):

    def dec(f):
        return f
    expected = getargspec(self.controller.index.__func__)
    actual = util.getargspec(dec(self.controller.index.__func__))
    assert expected == actual
    expected = getargspec(self.controller.static_index)
    actual = util.getargspec(dec(self.controller.static_index))
    assert expected == actual