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
def test_nonAuthoritativeMessage(self):
    """
        The L{RRHeader} instances created by L{Message} from a non-authoritative
        message are marked as not authoritative.
        """
    buf = BytesIO()
    answer = dns.RRHeader(payload=dns.Record_A('1.2.3.4', ttl=0))
    answer.encode(buf)
    message = dns.Message()
    message.fromStr(b'\x01\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00' + buf.getvalue())
    self.assertEqual(message.answers, [answer])
    self.assertFalse(message.answers[0].auth)