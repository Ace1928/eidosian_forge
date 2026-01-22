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
def test_NAPTR(self):
    """
        Test L{dns.Record_NAPTR} encode and decode.
        """
    naptrs = [(100, 10, b'u', b'sip+E2U', b'!^.*$!sip:information@domain.tld!', b''), (100, 50, b's', b'http+I2L+I2C+I2R', b'', b'_http._tcp.gatech.edu')]
    for order, preference, flags, service, regexp, replacement in naptrs:
        rin = dns.Record_NAPTR(order, preference, flags, service, regexp, replacement)
        e = BytesIO()
        rin.encode(e)
        e.seek(0, 0)
        rout = dns.Record_NAPTR()
        rout.decode(e)
        self.assertEqual(rin.order, rout.order)
        self.assertEqual(rin.preference, rout.preference)
        self.assertEqual(rin.flags, rout.flags)
        self.assertEqual(rin.service, rout.service)
        self.assertEqual(rin.regexp, rout.regexp)
        self.assertEqual(rin.replacement.name, rout.replacement.name)
        self.assertEqual(rin.ttl, rout.ttl)