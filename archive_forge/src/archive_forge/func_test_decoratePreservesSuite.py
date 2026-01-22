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
def test_decoratePreservesSuite(self):
    """
        Tests can be in non-standard suites. L{decorate} preserves the
        non-standard suites when it decorates the tests.
        """
    test = self.TestCase()
    suite = runner.DestructiveTestSuite([test])
    decorated = unittest.decorate(suite, unittest.TestDecorator)
    self.assertSuitesEqual(decorated, runner.DestructiveTestSuite([unittest.TestDecorator(test)]))