from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_pauseTwice(self):
    """
        Pauses on tasks should behave like a stack. If a task is paused twice,
        it needs to be resumed twice.
        """
    self.task.pause()
    self.scheduler.pump()
    self.assertEqual(self.work, [])
    self.task.pause()
    self.scheduler.pump()
    self.assertEqual(self.work, [])
    self.task.resume()
    self.scheduler.pump()
    self.assertEqual(self.work, [])
    self.task.resume()
    self.scheduler.pump()
    self.assertEqual(self.work, [1])