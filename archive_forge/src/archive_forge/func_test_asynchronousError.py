from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def test_asynchronousError(self):
    """
        Like L{test_asynchronousException}, but for a method which returns a
        Deferred failing with an L{pb.Error} subclass.
        """
    return self._exceptionTest('asynchronousError', AsynchronousError, False)