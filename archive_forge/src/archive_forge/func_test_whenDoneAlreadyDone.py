from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_whenDoneAlreadyDone(self):
    """
        L{CooperativeTask.whenDone} will return a L{defer.Deferred} that will
        succeed immediately if its iterator has already completed.
        """
    self.stopNext()
    self.scheduler.pump()
    results = []
    self.task.whenDone().addCallback(results.append)
    self.assertEqual(results, [self.task._iterator])