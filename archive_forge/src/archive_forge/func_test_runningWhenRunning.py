from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_runningWhenRunning(self):
    """
        L{Cooperator.running} reports C{True} when the L{Cooperator}
        is running.
        """
    c = task.Cooperator(started=False)
    c.start()
    self.addCleanup(c.stop)
    self.assertTrue(c.running)