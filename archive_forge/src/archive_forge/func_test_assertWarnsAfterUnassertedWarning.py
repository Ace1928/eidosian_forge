import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertWarnsAfterUnassertedWarning(self):
    """
        Warnings emitted before L{TestCase.assertWarns} is called do not get
        flushed and do not alter the behavior of L{TestCase.assertWarns}.
        """

    class TheWarning(Warning):
        pass

    def f(message):
        warnings.warn(message, category=TheWarning)
    f('foo')
    self.assertWarns(TheWarning, 'bar', __file__, f, 'bar')
    [warning] = self.flushWarnings([f])
    self.assertEqual(warning['message'], 'foo')