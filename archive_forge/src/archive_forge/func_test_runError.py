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
def test_runError(self) -> None:
    """
        Run a test, and encounter an error.
        """
    expectedCase = pyunitcases.PyUnitTest('test_error')
    result = self.workerRunTest(expectedCase)
    assert_that(result, matches_result(errors=has_length(1)))
    [(actualCase, failure)] = result.errors
    assert_that(expectedCase, equal_to(actualCase))
    assert_that(failure, isFailure(type=equal_to(Exception), value=equal_to(WorkerException('pyunit error')), frames=has_item(similarFrame('test_error', 'pyunitcases.py'))))