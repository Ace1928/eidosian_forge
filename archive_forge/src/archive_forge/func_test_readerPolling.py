from unittest import skipIf
from twisted.internet.error import ConnectionDone
from twisted.internet.posixbase import _ContinuousPolling
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_readerPolling(self):
    """
        Adding a reader causes its C{doRead} to be called every 1
        milliseconds.
        """
    reactor = Clock()
    poller = _ContinuousPolling(reactor)
    desc = Descriptor()
    poller.addReader(desc)
    self.assertEqual(desc.events, [])
    reactor.advance(1e-05)
    self.assertEqual(desc.events, ['read'])
    reactor.advance(1e-05)
    self.assertEqual(desc.events, ['read', 'read'])
    reactor.advance(1e-05)
    self.assertEqual(desc.events, ['read', 'read', 'read'])