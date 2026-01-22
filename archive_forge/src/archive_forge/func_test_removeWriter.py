from unittest import skipIf
from twisted.internet.error import ConnectionDone
from twisted.internet.posixbase import _ContinuousPolling
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_removeWriter(self):
    """
        Removing a writer stops the C{LoopingCall}.
        """
    poller = _ContinuousPolling(Clock())
    writer = object()
    poller.addWriter(writer)
    poller.removeWriter(writer)
    self.assertIsNone(poller._loop)
    self.assertEqual(poller._reactor.getDelayedCalls(), [])
    self.assertFalse(poller.isWriting(writer))