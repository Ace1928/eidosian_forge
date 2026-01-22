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
def test_simpleQuery(self):
    """
        Test content received after a query.
        """
    d = self.proto.query([dns.Query(b'foo')])
    self.assertEqual(len(self.proto.liveMessages.keys()), 1)
    m = dns.Message()
    m.id = next(iter(self.proto.liveMessages.keys()))
    m.answers = [dns.RRHeader(payload=dns.Record_A(address='1.2.3.4'))]

    def cb(result):
        self.assertEqual(result.answers[0].payload.dottedQuad(), '1.2.3.4')
    d.addCallback(cb)
    s = m.toStr()
    s = struct.pack('!H', len(s)) + s
    self.proto.dataReceived(s)
    return d