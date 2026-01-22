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
def test_addObservers(self):
    """
        Any log system observers L{TrialRunner.run} adds are removed by the
        time it returns.
        """
    originalCount = len(self._getObservers())
    self.runner.run(self.test)
    newCount = len(self._getObservers())
    self.assertEqual(newCount, originalCount)