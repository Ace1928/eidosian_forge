from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_pauseWhileDeferred(self):
    """
        C{pause()}ing a task while it is waiting on an outstanding
        L{defer.Deferred} should put the task into a state where the
        outstanding L{defer.Deferred} must be called back I{and} the task is
        C{resume}d before it will continue processing.
        """
    self.deferNext()
    self.scheduler.pump()
    self.assertEqual(len(self.work), 1)
    self.assertIsInstance(self.work[0], defer.Deferred)
    self.scheduler.pump()
    self.assertEqual(len(self.work), 1)
    self.task.pause()
    self.scheduler.pump()
    self.assertEqual(len(self.work), 1)
    self.task.resume()
    self.scheduler.pump()
    self.assertEqual(len(self.work), 1)
    self.work[0].callback('STUFF!')
    self.scheduler.pump()
    self.assertEqual(len(self.work), 2)
    self.assertEqual(self.work[1], 2)