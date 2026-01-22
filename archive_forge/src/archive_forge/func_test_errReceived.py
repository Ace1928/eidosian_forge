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
def test_errReceived(self):
    """
        L{LocalWorker.errReceived} logs the errors into its C{_errLog} log
        file.
        """
    localWorker = self.tidyLocalWorker(SpyDataLocalWorkerAMP(), FilePath(self.mktemp()), 'test.log')
    localWorker._errLog = BytesIO()
    data = b'The quick brown fox jumps over the lazy dog'
    localWorker.errReceived(data)
    self.assertEqual(data, localWorker._errLog.getvalue())