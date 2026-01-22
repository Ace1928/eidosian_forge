import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_brokenName(self):
    """
        If the exception type passed to C{assertRaises} does not have a
        C{__name__} then the context manager still manages to construct a
        descriptive string for it.
        """
    try:
        with self.assertRaises((ValueError, TypeError)):
            raise AttributeError()
    except self.failureException as exception:
        message = str(exception)
        valueError = 'ValueError' not in message
        typeError = 'TypeError' not in message
        errors = []
        if valueError:
            errors.append('expected ValueError in exception message')
        if typeError:
            errors.append('expected TypeError in exception message')
        if errors:
            self.fail('; '.join(errors), f'message = {message}')
    else:
        self.fail('Mismatched exception type should have caused test failure.')