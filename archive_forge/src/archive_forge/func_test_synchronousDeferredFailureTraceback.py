import types
from typing_extensions import NoReturn
from twisted.internet.defer import (
from twisted.internet.task import Clock
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_synchronousDeferredFailureTraceback(self) -> None:
    """
        When a Deferred is awaited upon that has already failed with a Failure
        that has a traceback, both the place that the synchronous traceback
        comes from and the awaiting line are shown in the traceback.
        """

    def raises() -> None:
        raise SampleException()
    it = maybeDeferred(raises)

    async def doomed() -> None:
        return await it
    failure = self.failureResultOf(Deferred.fromCoroutine(doomed()))
    self.assertIn(', in doomed\n', failure.getTraceback())
    self.assertIn(', in raises\n', failure.getTraceback())