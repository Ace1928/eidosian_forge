import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertRaisesContextNoException(self):
    """
        If C{assertRaises} is used to create a context manager and no exception
        is raised from the body of the C{with} statement then the C{with}
        statement raises C{failureException} describing the lack of exception.
        """
    try:
        with self.assertRaises(ValueError):
            pass
    except self.failureException as exception:
        message = str(exception)
        self.assertEqual(message, 'ValueError not raised (None returned)')
    else:
        self.fail('Non-exception result should have caused test failure.')