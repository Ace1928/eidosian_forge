from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def test_deferredSecurity(self):
    """
        Test that a Deferred which fails with a L{pb.Error} which is not
        also a L{pb.Jellyable} is treated in the same way as a synchronously
        raised exception of the same type.
        """

    def failureDeferredSecurity(fail):
        fail.trap(SecurityError)
        self.assertNotIsInstance(fail.type, str)
        self.assertIsInstance(fail.value, fail.type)
        return 43000
    return self._testImpl('deferredSecurity', 43000, failureDeferredSecurity)