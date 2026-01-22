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
def test_revertDuringTest(self):
    """
        C{patch()} return a L{monkey.MonkeyPatcher} object that can be used to
        restore the original values before the end of the test.
        """
    patch = self.test.patch(self, 'objectToPatch', self.patchedValue)
    patch.restore()
    self.assertEqual(self.objectToPatch, self.originalValue)