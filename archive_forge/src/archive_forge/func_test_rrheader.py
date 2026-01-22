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
def test_rrheader(self):
    """
        Two L{dns.RRHeader} instances compare equal if and only if they have
        the same name, type, class, time to live, payload, and authoritative
        bit.
        """
    self._equalityTest(dns.RRHeader(b'example.com', payload=dns.Record_A('1.2.3.4')), dns.RRHeader(b'example.com', payload=dns.Record_A('1.2.3.4')), dns.RRHeader(b'example.org', payload=dns.Record_A('1.2.3.4')))
    self._equalityTest(dns.RRHeader(b'example.com', payload=dns.Record_A('1.2.3.4')), dns.RRHeader(b'example.com', payload=dns.Record_A('1.2.3.4')), dns.RRHeader(b'example.com', payload=dns.Record_A('1.2.3.5')))
    self._equalityTest(dns.RRHeader(b'example.com', dns.A), dns.RRHeader(b'example.com', dns.A), dns.RRHeader(b'example.com', dns.MX))
    self._equalityTest(dns.RRHeader(b'example.com', cls=dns.IN, payload=dns.Record_A('1.2.3.4')), dns.RRHeader(b'example.com', cls=dns.IN, payload=dns.Record_A('1.2.3.4')), dns.RRHeader(b'example.com', cls=dns.CS, payload=dns.Record_A('1.2.3.4')))
    self._equalityTest(dns.RRHeader(b'example.com', ttl=60, payload=dns.Record_A('1.2.3.4')), dns.RRHeader(b'example.com', ttl=60, payload=dns.Record_A('1.2.3.4')), dns.RRHeader(b'example.com', ttl=120, payload=dns.Record_A('1.2.3.4')))
    self._equalityTest(dns.RRHeader(b'example.com', auth=1, payload=dns.Record_A('1.2.3.4')), dns.RRHeader(b'example.com', auth=1, payload=dns.Record_A('1.2.3.4')), dns.RRHeader(b'example.com', auth=0, payload=dns.Record_A('1.2.3.4')))