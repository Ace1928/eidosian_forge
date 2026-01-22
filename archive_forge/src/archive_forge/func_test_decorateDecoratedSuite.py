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
def test_decorateDecoratedSuite(self):
    """
        Calling L{decorate} on a test suite with already-decorated tests
        decorates all of the tests in the suite again.
        """
    test = self.TestCase()
    decoratedTest = unittest.decorate(test, unittest.TestDecorator)
    redecoratedTest = unittest.decorate(decoratedTest, unittest.TestDecorator)
    self.assertTestsEqual(redecoratedTest, unittest.TestDecorator(decoratedTest))