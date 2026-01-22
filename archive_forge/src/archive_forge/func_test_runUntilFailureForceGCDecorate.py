import os
import pdb
import sys
import unittest as pyunit
from io import StringIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted import plugin
from twisted.internet import defer
from twisted.plugins import twisted_trial
from twisted.python import failure, log, reflect
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedAny
from twisted.scripts import trial
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import _ForceGarbageCollectionDecorator
from twisted.trial.itrial import IReporter, ITestCase
def test_runUntilFailureForceGCDecorate(self):
    """
        C{runUntilFailure} applies the force-gc decoration after the standard
        L{ITestCase} decoration, but only one time.
        """
    decorated = []

    def decorate(test, interface):
        decorated.append((test, interface))
        return test
    self.patch(unittest, 'decorate', decorate)
    self.runner._forceGarbageCollection = True
    result = self.runner.runUntilFailure(self.test)
    self.assertEqual(result.testsRun, 1)
    self.assertEqual(len(decorated), 2)
    self.assertEqual(decorated, [(self.test, ITestCase), (self.test, _ForceGarbageCollectionDecorator)])