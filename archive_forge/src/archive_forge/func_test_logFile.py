import os
import sys
from functools import partial
from io import StringIO
from os.path import sep
from typing import Callable, List, Set
from unittest import TestCase as PyUnitTestCase
from zope.interface import implementer, verify
from attrs import Factory, assoc, define, field
from hamcrest import (
from hamcrest.core.core.allof import AllOf
from hypothesis import given
from hypothesis.strategies import booleans, sampled_from
from twisted.internet import interfaces
from twisted.internet.base import ReactorBase
from twisted.internet.defer import CancelledError, Deferred, succeed
from twisted.internet.error import ProcessDone
from twisted.internet.protocol import ProcessProtocol, Protocol
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.internet.testing import MemoryReactorClock
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
from twisted.trial._dist import _WORKER_AMP_STDIN
from twisted.trial._dist.distreporter import DistReporter
from twisted.trial._dist.disttrial import DistTrialRunner, WorkerPool, WorkerPoolConfig
from twisted.trial._dist.functional import (
from twisted.trial._dist.worker import LocalWorker, RunResult, Worker, WorkerAction
from twisted.trial.reporter import (
from twisted.trial.runner import ErrorHolder, TrialSuite
from twisted.trial.unittest import SynchronousTestCase, TestCase
from ...test import erroneous, sample
from .matchers import matches_result
@given(booleans(), sampled_from(['out.log', f'subdir{sep}out.log']))
def test_logFile(self, absolute: bool, logFile: str) -> None:
    """
        L{WorkerPool.start} creates a L{StartedWorkerPool} configured with a
        log file based on the L{WorkerPoolConfig.logFile}.
        """
    if absolute:
        logFile = self.parent.path + sep + logFile
    config = assoc(self.config, logFile=logFile)
    if absolute:
        matches = equal_to(logFile)
    else:
        matches = AllOf(starts_with(config.workingDirectory.path), ends_with(sep + logFile))
    pool = WorkerPool(config)
    started = self.successResultOf(pool.start(CountingReactor([])))
    assert_that(started.testLog.name, matches)