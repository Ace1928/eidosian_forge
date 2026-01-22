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
def loadSuite(self, suite):
    """
        Load tests from the given test case class and create a new reporter to
        use for running it.
        """
    self.loader = pyunit.TestLoader()
    self.suite = self.loader.loadTestsFromTestCase(suite)
    self.reporter = reporter.TestResult()