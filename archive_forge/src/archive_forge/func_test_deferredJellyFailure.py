from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def test_deferredJellyFailure(self):
    """
        Test that a Deferred which fails with a L{pb.Error} is treated in
        the same way as a synchronously raised L{pb.Error}.
        """

    def failureDeferredJelly(fail):
        fail.trap(JellyError)
        self.assertNotIsInstance(fail.type, str)
        self.assertIsInstance(fail.value, fail.type)
        return 430
    return self._testImpl('deferredJelly', 430, failureDeferredJelly)