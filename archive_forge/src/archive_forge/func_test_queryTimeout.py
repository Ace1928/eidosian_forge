import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
def test_queryTimeout(self):
    """
        Test that query timeouts after some seconds.
        """
    d = self.proto.query([dns.Query(b'foo')])
    self.assertEqual(len(self.proto.liveMessages), 1)
    self.clock.advance(60)
    self.assertFailure(d, dns.DNSQueryTimeoutError)
    self.assertEqual(len(self.proto.liveMessages), 0)
    return d