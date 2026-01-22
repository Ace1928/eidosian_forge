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
def test_successfulIsReported(self):
    """
        Test that when a successful test is run, it is reported as a success,
        and not as any other kind of result.
        """
    test = self.__class__('test_successful')
    test.run(self.result)
    self.assertSuccessful(test, self.result)