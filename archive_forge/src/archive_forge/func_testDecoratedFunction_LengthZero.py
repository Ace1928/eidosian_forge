import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testDecoratedFunction_LengthZero(self):

    @util.positional(0)
    def fn(kwonly=1):
        return [kwonly]
    self.assertEquals([1], fn())
    self.assertEquals([2], fn(kwonly=2))
    self.assertRaisesWithRegexpMatch(TypeError, 'fn\\(\\) takes at most 0 positional arguments \\(1 given\\)', fn, 1)