from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_runningWhenStopped(self):
    """
        L{Cooperator.running} reports C{False} after the L{Cooperator}
        has been stopped.
        """
    c = task.Cooperator(started=False)
    c.start()
    c.stop()
    self.assertFalse(c.running)