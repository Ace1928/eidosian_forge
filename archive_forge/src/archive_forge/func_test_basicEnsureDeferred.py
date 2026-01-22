import types
from typing_extensions import NoReturn
from twisted.internet.defer import (
from twisted.internet.task import Clock
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_basicEnsureDeferred(self) -> None:
    """
        L{ensureDeferred} allows a function to C{await} on a L{Deferred}.
        """

    async def run() -> str:
        d = succeed('foo')
        res = await d
        return res
    d = ensureDeferred(run())
    res = self.successResultOf(d)
    self.assertEqual(res, 'foo')