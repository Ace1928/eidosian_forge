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
def tidyLocalWorker(self, *args, **kwargs):
    """
        Create a L{LocalWorker}, connect it to a transport, and ensure
        its log files are closed.

        @param args: See L{LocalWorker}

        @param kwargs: See L{LocalWorker}

        @return: a L{LocalWorker} instance
        """
    worker = LocalWorker(*args, **kwargs)
    worker.makeConnection(FakeTransport())
    self.addCleanup(worker._outLog.close)
    self.addCleanup(worker._errLog.close)
    return worker