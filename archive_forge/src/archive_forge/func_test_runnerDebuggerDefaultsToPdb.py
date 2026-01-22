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
def test_runnerDebuggerDefaultsToPdb(self):
    """
        Trial uses pdb if no debugger is specified by `--debugger`
        """
    self.parseOptions(['--debug', 'twisted.trial.test.sample'])
    pdbrcFile = FilePath('pdbrc')
    pdbrcFile.touch()
    self.runcall_called = False

    def runcall(pdb, suite, result):
        self.runcall_called = True
    self.patch(pdb.Pdb, 'runcall', runcall)
    self.runSampleSuite(self.getRunner())
    self.assertTrue(self.runcall_called)