import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertWarnsOnCall(self):
    """
        Test assertWarns works on instance with C{__call__} method.
        """

    class Warn:

        def __call__(self, a):
            warnings.warn('Egg deprecated', category=DeprecationWarning)
            return a
    w = Warn()
    r = self.assertWarns(DeprecationWarning, 'Egg deprecated', __file__, w, 321)
    self.assertEqual(r, 321)
    r = self.assertWarns(DeprecationWarning, 'Egg deprecated', __file__, w, 321)
    self.assertEqual(r, 321)