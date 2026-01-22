from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_runningWhenNotStarted(self):
    """
        L{Cooperator.running} reports C{False} if the L{Cooperator}
        has not been started.
        """
    c = task.Cooperator(started=False)
    self.assertFalse(c.running)