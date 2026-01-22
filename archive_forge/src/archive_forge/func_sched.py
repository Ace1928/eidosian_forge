from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def sched(f):
    calls[0] = FakeDelayedCall(f)
    return calls[0]