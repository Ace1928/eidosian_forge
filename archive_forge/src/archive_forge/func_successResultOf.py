import inspect
import os
import sys
import tempfile
import types
import unittest as pyunit
import warnings
from dis import findlinestarts as _findlinestarts
from typing import (
from unittest import SkipTest
from attrs import frozen
from typing_extensions import ParamSpec
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python import failure, log, monkey
from twisted.python.deprecate import (
from twisted.python.reflect import fullyQualifiedName
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import itrial, util
def successResultOf(self, deferred: Union[Coroutine[Deferred[T], Any, T], Generator[Deferred[T], Any, T], Deferred[T]]) -> T:
    """
        Return the current success result of C{deferred} or raise
        C{self.failureException}.

        @param deferred: A L{Deferred<twisted.internet.defer.Deferred>} or
            I{coroutine} which has a success result.

            For a L{Deferred<twisted.internet.defer.Deferred>} this means
            L{Deferred.callback<twisted.internet.defer.Deferred.callback>} or
            L{Deferred.errback<twisted.internet.defer.Deferred.errback>} has
            been called on it and it has reached the end of its callback chain
            and the last callback or errback returned a
            non-L{failure.Failure}.

            For a I{coroutine} this means all awaited values have a success
            result.

        @raise SynchronousTestCase.failureException: If the
            L{Deferred<twisted.internet.defer.Deferred>} has no result or has a
            failure result.

        @return: The result of C{deferred}.
        """
    deferred = ensureDeferred(deferred)
    results: List[Union[T, failure.Failure]] = []
    deferred.addBoth(results.append)
    if not results:
        self.fail('Success result expected on {!r}, found no result instead'.format(deferred))
    result = results[0]
    if isinstance(result, failure.Failure):
        self.fail('Success result expected on {!r}, found failure result instead:\n{}'.format(deferred, result.getTraceback()))
    return result