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
def test_soa(self):
    """
        Two L{dns.Record_SOA} instances compare equal if and only if they have
        the same mname, rname, serial, refresh, minimum, expire, retry, and
        ttl.
        """
    self._equalityTest(dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 20, 30), dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 20, 30), dns.Record_SOA(b'xname', b'rname', 123, 456, 789, 10, 20, 30))
    self._equalityTest(dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 20, 30), dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 20, 30), dns.Record_SOA(b'mname', b'xname', 123, 456, 789, 10, 20, 30))
    self._equalityTest(dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 20, 30), dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 20, 30), dns.Record_SOA(b'mname', b'rname', 1, 456, 789, 10, 20, 30))
    self._equalityTest(dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 20, 30), dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 20, 30), dns.Record_SOA(b'mname', b'rname', 123, 1, 789, 10, 20, 30))
    self._equalityTest(dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 20, 30), dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 20, 30), dns.Record_SOA(b'mname', b'rname', 123, 456, 1, 10, 20, 30))
    self._equalityTest(dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 20, 30), dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 20, 30), dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 1, 20, 30))
    self._equalityTest(dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 20, 30), dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 20, 30), dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 1, 30))
    self._equalityTest(dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 20, 30), dns.Record_SOA(b'mname', b'rname', 123, 456, 789, 10, 20, 30), dns.Record_SOA(b'mname', b'xname', 123, 456, 789, 10, 20, 1))