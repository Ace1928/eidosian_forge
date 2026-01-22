import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertWarnsMultipleWarnings(self):
    """
        C{assertWarns} does not raise an exception if the function it is passed
        triggers the same warning more than once.
        """

    def deprecated():
        warnings.warn('Woo deprecated', category=PendingDeprecationWarning)

    def f():
        deprecated()
        deprecated()
    self.assertWarns(PendingDeprecationWarning, 'Woo deprecated', __file__, f)