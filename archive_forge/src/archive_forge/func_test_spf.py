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
def test_spf(self):
    """
        L{dns.Record_SPF} instances compare equal if and only if they have the
        same data and ttl.
        """
    self._equalityTest(dns.Record_SPF('foo', 'bar', ttl=10), dns.Record_SPF('foo', 'bar', ttl=10), dns.Record_SPF('foo', 'bar', 'baz', ttl=10))
    self._equalityTest(dns.Record_SPF('foo', 'bar', ttl=10), dns.Record_SPF('foo', 'bar', ttl=10), dns.Record_SPF('bar', 'foo', ttl=10))
    self._equalityTest(dns.Record_SPF('foo', 'bar', ttl=10), dns.Record_SPF('foo', 'bar', ttl=10), dns.Record_SPF('foo', 'bar', ttl=100))