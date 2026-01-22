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
def test_addCleanupCalledIfSetUpFails(self):
    """
        Callables added with C{addCleanup} are run even if setUp fails.
        """
    self.test.setUp = self.test.brokenSetUp
    self.test.addCleanup(self.test.append, 'foo')
    self.test.run(self.result)
    self.assertEqual(['setUp', 'foo'], self.test.log)