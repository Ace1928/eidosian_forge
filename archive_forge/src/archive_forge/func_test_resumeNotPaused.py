from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_resumeNotPaused(self):
    """
        L{CooperativeTask.resume} should raise a L{TaskNotPaused} exception if
        it was not paused; e.g. if L{CooperativeTask.pause} was not invoked
        more times than L{CooperativeTask.resume} on that object.
        """
    self.assertRaises(task.NotPaused, self.task.resume)
    self.task.pause()
    self.task.resume()
    self.assertRaises(task.NotPaused, self.task.resume)