import gc
import sys
import unittest as pyunit
import weakref
from io import StringIO
from twisted.internet import defer, reactor
from twisted.python.compat import _PYPY
from twisted.python.reflect import namedAny
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import (
from twisted.trial.test import erroneous
from twisted.trial.test.test_suppression import SuppressionMixin
def test_multipleErrorsReported(self):
    """
        If more than one cleanup fails, then the test should fail with more
        than one error.
        """
    self.test.addCleanup(self.test.fail, 'foo')
    self.test.addCleanup(self.test.fail, 'bar')
    self.test.run(self.result)
    self.assertEqual(['setUp', 'runTest', 'tearDown'], self.test.log)
    self.assertEqual(2, len(self.result.errors))
    [(test1, error1), (test2, error2)] = self.result.errors
    self.assertEqual(test1, self.test)
    self.assertEqual(test2, self.test)
    self.assertEqual(error1.getErrorMessage(), 'bar')
    self.assertEqual(error2.getErrorMessage(), 'foo')