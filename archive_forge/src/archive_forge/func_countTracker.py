from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def countTracker(possibleCount):
    deferredCounts.append(possibleCount)
    if len(deferredCounts) == 1:
        return d
    else:
        return None