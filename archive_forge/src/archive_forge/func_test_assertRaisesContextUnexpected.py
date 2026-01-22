import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertRaisesContextUnexpected(self):
    """
        If C{assertRaises} is used to create a context manager and the wrong
        exception type is raised from the body of the C{with} statement then
        the C{with} statement raises C{failureException} describing the
        mismatch.
        """
    try:
        with self.assertRaises(ValueError):
            raise TypeError('marker')
    except self.failureException as exception:
        message = str(exception)
        expected = '{type} raised instead of ValueError:\n Traceback'.format(type=fullyQualifiedName(TypeError))
        self.assertTrue(message.startswith(expected), 'Exception message did not begin with expected information: {}'.format(message))
    else:
        self.fail('Mismatched exception type should have caused test failure.')