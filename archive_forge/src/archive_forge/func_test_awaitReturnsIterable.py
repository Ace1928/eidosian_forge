import types
from typing_extensions import NoReturn
from twisted.internet.defer import (
from twisted.internet.task import Clock
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_awaitReturnsIterable(self) -> None:
    """
        C{Deferred.__await__} returns an iterable.
        """
    d: Deferred[None] = Deferred()
    awaitedDeferred = d.__await__()
    self.assertEqual(awaitedDeferred, iter(awaitedDeferred))