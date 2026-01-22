from unittest import skipIf
from twisted.internet.error import ConnectionDone
from twisted.internet.posixbase import _ContinuousPolling
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_removeUnknown(self):
    """
        Removing unknown readers and writers silently does nothing.
        """
    poller = _ContinuousPolling(Clock())
    poller.removeWriter(object())
    poller.removeReader(object())