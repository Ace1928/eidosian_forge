from unittest import skipIf
from twisted.internet.error import ConnectionDone
from twisted.internet.posixbase import _ContinuousPolling
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_removeAll(self):
    """
        L{_ContinuousPolling.removeAll} removes all descriptors and returns
        the readers and writers.
        """
    poller = _ContinuousPolling(Clock())
    reader = object()
    writer = object()
    both = object()
    poller.addReader(reader)
    poller.addReader(both)
    poller.addWriter(writer)
    poller.addWriter(both)
    removed = poller.removeAll()
    self.assertEqual(poller.getReaders(), [])
    self.assertEqual(poller.getWriters(), [])
    self.assertEqual(len(removed), 3)
    self.assertEqual(set(removed), {reader, writer, both})