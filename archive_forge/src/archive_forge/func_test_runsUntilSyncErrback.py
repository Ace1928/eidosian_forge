from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_runsUntilSyncErrback(self):
    """
        L{task.react} returns quickly if the L{defer.Deferred} returned by the
        function it is passed has already been errbacked at the time it is
        returned.
        """

    class ExpectedException(Exception):
        pass

    async def main(reactor):
        return await defer.fail(ExpectedException())
    r = _FakeReactor()
    exitError = self.assertRaises(SystemExit, task.react, main, _reactor=r)
    self.assertEqual(1, exitError.code)
    self.assertEqual(r.seconds(), 0)
    errors = self.flushLoggedErrors(ExpectedException)
    self.assertEqual(len(errors), 1)