from unittest import skipIf
from twisted.internet.error import ConnectionDone
from twisted.internet.posixbase import _ContinuousPolling
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_connectionLostOnRead(self):
    """
        If a C{doRead} returns a value indicating disconnection,
        C{connectionLost} is called on it.
        """
    reactor = Clock()
    poller = _ContinuousPolling(reactor)
    desc = Descriptor()
    desc.doRead = lambda: ConnectionDone()
    poller.addReader(desc)
    self.assertEqual(desc.events, [])
    reactor.advance(0.001)
    self.assertEqual(desc.events, ['lost'])