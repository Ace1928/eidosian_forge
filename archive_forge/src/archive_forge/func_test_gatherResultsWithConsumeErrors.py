from __future__ import annotations
import contextvars
import functools
import gc
import re
import traceback
import types
import unittest as pyunit
import warnings
import weakref
from asyncio import (
from typing import (
from hamcrest import assert_that, empty, equal_to
from hypothesis import given
from hypothesis.strategies import integers
from typing_extensions import assert_type
from twisted.internet import defer, reactor
from twisted.internet.defer import (
from twisted.internet.task import Clock
from twisted.python import log
from twisted.python.compat import _PYPY
from twisted.python.failure import Failure
from twisted.trial import unittest
def test_gatherResultsWithConsumeErrors(self) -> None:
    """
        If a L{Deferred} in the list passed to L{gatherResults} fires with a
        failure and C{consumerErrors} is C{True}, the failure is converted to a
        L{None} result on that L{Deferred}.
        """
    dgood = defer.succeed(1)
    dbad = defer.fail(RuntimeError('oh noes'))
    d = defer.gatherResults([dgood, dbad], consumeErrors=True)
    unconsumedErrors: List[Failure] = []
    dbad.addErrback(unconsumedErrors.append)
    gatheredErrors: List[Failure] = []
    d.addErrback(gatheredErrors.append)
    self.assertEqual((len(unconsumedErrors), len(gatheredErrors)), (0, 1))
    self.assertIsInstance(gatheredErrors[0].value, defer.FirstError)
    firstError = gatheredErrors[0].value.subFailure
    self.assertIsInstance(firstError.value, RuntimeError)