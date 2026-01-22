from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def test_asynchronousException(self):
    """
        Test that a Deferred returned by a remote method which already has a
        Failure correctly has that error passed back to the calling side.
        """
    return self._exceptionTest('asynchronousException', AsynchronousException, True)