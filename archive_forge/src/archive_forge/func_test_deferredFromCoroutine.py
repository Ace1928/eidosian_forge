import types
from typing_extensions import NoReturn
from twisted.internet.defer import (
from twisted.internet.task import Clock
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_deferredFromCoroutine(self) -> None:
    """
        L{Deferred.fromCoroutine} will turn a coroutine into a L{Deferred}.
        """

    async def run() -> str:
        d = succeed('bar')
        await d
        res = await run2()
        return res

    async def run2() -> str:
        d = succeed('foo')
        res = await d
        return res
    r = run()
    self.assertIsInstance(r, types.CoroutineType)
    d = Deferred.fromCoroutine(r)
    self.assertIsInstance(d, Deferred)
    res = self.successResultOf(d)
    self.assertEqual(res, 'foo')