from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_whenDoneStop(self):
    """
        L{CooperativeTask.whenDone} returns a L{defer.Deferred} that fails with
        L{TaskStopped} when the C{stop} method is called on that
        L{CooperativeTask}.
        """
    deferred1 = self.task.whenDone()
    errors = []
    deferred1.addErrback(errors.append)
    self.task.stop()
    self.assertEqual(len(errors), 1)
    self.assertEqual(errors[0].check(task.TaskStopped), task.TaskStopped)