from unittest import skipIf
from twisted.internet.error import ConnectionDone
from twisted.internet.posixbase import _ContinuousPolling
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_addWriter(self):
    """
        Adding a writer when there was previously no writer starts up a
        C{LoopingCall}.
        """
    poller = _ContinuousPolling(Clock())
    self.assertIsNone(poller._loop)
    writer = object()
    self.assertFalse(poller.isWriting(writer))
    poller.addWriter(writer)
    self.assertIsNotNone(poller._loop)
    self.assertTrue(poller._loop.running)
    self.assertIs(poller._loop.clock, poller._reactor)
    self.assertTrue(poller.isWriting(writer))