import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertWarnsRegistryClean(self):
    """
        Test that assertWarns cleans the warning registry, so the warning is
        not swallowed the second time.
        """

    def deprecated(a):
        warnings.warn('Woo deprecated', category=DeprecationWarning)
        return a
    r1 = self.assertWarns(DeprecationWarning, 'Woo deprecated', __file__, deprecated, 123)
    self.assertEqual(r1, 123)
    r2 = self.assertWarns(DeprecationWarning, 'Woo deprecated', __file__, deprecated, 321)
    self.assertEqual(r2, 321)