from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_pauseStopResume(self):
    """
        C{resume()}ing a paused, stopped task should be a no-op; it should not
        raise an exception, because it's paused, but neither should it actually
        do more work from the task.
        """
    self.task.pause()
    self.task.stop()
    self.task.resume()
    self.scheduler.pump()
    self.assertEqual(self.work, [])