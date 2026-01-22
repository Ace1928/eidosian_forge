from unittest import skipIf
from twisted.internet.error import ConnectionDone
from twisted.internet.posixbase import _ContinuousPolling
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_writerPolling(self):
    """
        Adding a writer causes its C{doWrite} to be called every 1
        milliseconds.
        """
    reactor = Clock()
    poller = _ContinuousPolling(reactor)
    desc = Descriptor()
    poller.addWriter(desc)
    self.assertEqual(desc.events, [])
    reactor.advance(0.001)
    self.assertEqual(desc.events, ['write'])
    reactor.advance(0.001)
    self.assertEqual(desc.events, ['write', 'write'])
    reactor.advance(0.001)
    self.assertEqual(desc.events, ['write', 'write', 'write'])