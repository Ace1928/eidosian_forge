from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_runsUntilAsyncCallback(self):
    """
        L{task.react} runs the reactor until the L{Deferred} returned by the
        function it is passed is called back, then stops it.
        """
    timePassed = []

    async def main(reactor):
        finished = defer.Deferred()
        reactor.callLater(1, timePassed.append, True)
        reactor.callLater(2, finished.callback, None)
        return await finished
    r = _FakeReactor()
    exitError = self.assertRaises(SystemExit, task.react, main, _reactor=r)
    self.assertEqual(0, exitError.code)
    self.assertEqual(timePassed, [True])
    self.assertEqual(r.seconds(), 2)