import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertEqual_incomparable(self):
    apple = ComparisonError()
    orange = ['orange']
    try:
        self.assertEqual(apple, orange)
    except self.failureException:
        self.fail('Fail raised when ValueError ought to have been raised.')
    except ValueError:
        pass
    else:
        self.fail('Comparing {!r} and {!r} should have raised an exception'.format(apple, orange))