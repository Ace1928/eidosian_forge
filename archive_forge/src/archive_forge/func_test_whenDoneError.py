from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_whenDoneError(self):
    """
        L{CooperativeTask.whenDone} returns a L{defer.Deferred} that will fail
        when the iterable's C{next} method raises an exception, with that
        exception.
        """
    deferred1 = self.task.whenDone()
    results = []
    deferred1.addErrback(results.append)
    self.dieNext()
    self.scheduler.pump()
    self.assertEqual(len(results), 1)
    self.assertEqual(results[0].check(UnhandledException), UnhandledException)