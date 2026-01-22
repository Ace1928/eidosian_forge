import os
from io import BytesIO, StringIO
from typing import Type
from unittest import TestCase as PyUnitTestCase
from zope.interface.verify import verifyObject
from hamcrest import assert_that, equal_to, has_item, has_length
from twisted.internet.defer import Deferred, fail
from twisted.internet.error import ConnectionLost, ProcessDone
from twisted.internet.interfaces import IAddress, ITransport
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.trial._dist import managercommands
from twisted.trial._dist.worker import (
from twisted.trial.reporter import TestResult
from twisted.trial.test import pyunitcases, skipping
from twisted.trial.unittest import TestCase, makeTodo
from .matchers import isFailure, matches_result, similarFrame
def test_newlineStyle(self):
    """
        L{LocalWorker} writes the log data with local newlines.
        """
    amp = SpyDataLocalWorkerAMP()
    tempDir = FilePath(self.mktemp())
    tempDir.makedirs()
    logPath = tempDir.child('test.log')
    with open(logPath.path, 'wt', encoding='utf-8') as logFile:
        worker = LocalWorker(amp, tempDir, logFile)
        worker.makeConnection(FakeTransport())
        self.addCleanup(worker._outLog.close)
        self.addCleanup(worker._errLog.close)
        expected = 'Here comes the ☉!'
        amp.testWrite(expected)
    self.assertEqual(expected + os.linesep, logPath.getContent().decode('utf-8'))