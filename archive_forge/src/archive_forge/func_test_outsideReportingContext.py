from __future__ import annotations
from typing import Sized
from unittest import TestCase
from hamcrest import assert_that, equal_to, has_length
from hamcrest.core.matcher import Matcher
from twisted.internet.defer import Deferred
from twisted.test.iosim import connectedServerAndClient
from twisted.trial._dist.worker import LocalWorkerAMP, WorkerProtocol
from twisted.trial.reporter import TestResult
from twisted.trial.test import erroneous, pyunitcases, sample, skipping
from twisted.trial.unittest import SynchronousTestCase
from .matchers import matches_result
def test_outsideReportingContext(self) -> None:
    """
        L{WorkerReporter}'s implementation of test result methods raise
        L{ValueError} when called outside of the
        L{WorkerReporter.gatherReportingResults} context manager.
        """
    worker, local, pump = connectedServerAndClient(LocalWorkerAMP, WorkerProtocol)
    case = sample.FooTest('test_foo')
    with self.assertRaises(ValueError):
        worker._result.addSuccess(case)