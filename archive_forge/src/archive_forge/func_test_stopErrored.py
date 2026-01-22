from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_stopErrored(self):
    """
        C{stop()}ping a L{CooperativeTask} whose iterator has encountered an
        error should raise L{TaskFailed}.
        """
    self.dieNext()
    self.scheduler.pump()
    self.assertRaises(task.TaskFailed, self.task.stop)