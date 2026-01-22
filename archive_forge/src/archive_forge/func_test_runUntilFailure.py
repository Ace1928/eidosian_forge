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
def test_runUntilFailure(self):
    """
        L{DistTrialRunner} can run in C{untilFailure} mode where it will run
        the given tests until they fail.
        """
    stream = StringIO()
    case = erroneous.EventuallyFailingTestCase('test_it')
    runner = self.getRunner(stream=stream)
    d = runner.runAsync(case, untilFailure=True)
    result = self.successResultOf(d)
    self.assertEqual(5, case.n)
    self.assertFalse(result.wasSuccessful())
    output = stream.getvalue()
    self.assertEqual(output.count('PASSED'), case.n - 1, 'expected to see PASSED in output')
    self.assertIn('FAIL', output)
    for i in range(1, 6):
        self.assertIn(f'Test Pass {i}', output)
    self.assertEqual(output.count('Ran 1 tests in'), case.n, 'expected to see per-iteration test count in output')