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
def test_resourceRecordHeader(self):
    """
        L{dns.RRHeader.encode} encodes the record header's information and
        writes it to the file-like object passed to it and
        L{dns.RRHeader.decode} reads from a file-like object to re-construct a
        L{dns.RRHeader} instance.
        """
    f = BytesIO()
    dns.RRHeader(b'test.org', 3, 4, 17).encode(f)
    f.seek(0, 0)
    result = dns.RRHeader()
    result.decode(f)
    self.assertEqual(result.name, dns.Name(b'test.org'))
    self.assertEqual(result.type, 3)
    self.assertEqual(result.cls, 4)
    self.assertEqual(result.ttl, 17)