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
def test_decorateTestSuite(self):
    """
        Calling L{decorate} on a test suite will return a test suite with
        each test decorated with the provided decorator.
        """
    test = self.TestCase()
    suite = unittest.TestSuite([test])
    decoratedTest = unittest.decorate(suite, unittest.TestDecorator)
    self.assertSuitesEqual(decoratedTest, unittest.TestSuite([unittest.TestDecorator(test)]))