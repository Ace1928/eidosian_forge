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
def test_cleanupRunsOnce(self):
    """
        A function registered as a cleanup is run once.
        """
    cleanups = []
    self.test.addCleanup(lambda: cleanups.append(stage))
    stage = 'first'
    self.test.run(self.result)
    stage = 'second'
    self.test.run(self.result)
    self.assertEqual(cleanups, ['first'])