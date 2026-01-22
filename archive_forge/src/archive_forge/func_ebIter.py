from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def ebIter(self, err):
    err.trap(task.SchedulerStopped)
    return self.RESULT