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
@pyunit.skipIf(_PYPY, 'GC works differently on PyPy.')
def test_doesntBleed(self):
    """
        Forcing garbage collection in the test should mean that there are
        no unreachable cycles immediately after the test completes.
        """
    result = reporter.TestResult()
    self.test1(result)
    self.flushLoggedErrors()
    n = len(gc.garbage)
    self.assertEqual(n, 0, 'unreachable cycle still existed')
    x = self.flushLoggedErrors()
    self.assertEqual(len(x), 0, 'Errors logged after gc.collect')