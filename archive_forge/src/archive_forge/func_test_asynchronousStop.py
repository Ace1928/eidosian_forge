from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_asynchronousStop(self):
    """
        L{task.react} handles when the reactor is stopped and the
        returned L{Deferred} doesn't fire.
        """

    async def main(reactor):
        reactor.callLater(1, reactor.stop)
        return await defer.Deferred()
    r = _FakeReactor()
    exitError = self.assertRaises(SystemExit, task.react, main, [], _reactor=r)
    self.assertEqual(0, exitError.code)