from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def failureNoSuch(fail):
    fail.trap(pb.NoSuchMethod)
    self.compare(fail.traceback, 'Traceback unavailable\n')
    return 42000