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
def test_successivePatches(self):
    """
        Successive patches are applied and reverted just like a single patch.
        """
    self.test.patch(self, 'objectToPatch', self.patchedValue)
    self.assertEqual(self.objectToPatch, self.patchedValue)
    self.test.patch(self, 'objectToPatch', 'second value')
    self.assertEqual(self.objectToPatch, 'second value')
    self.test.run(reporter.Reporter())
    self.assertEqual(self.objectToPatch, self.originalValue)