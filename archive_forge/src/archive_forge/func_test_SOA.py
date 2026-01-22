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
def test_SOA(self):
    """
        The byte stream written by L{dns.Record_SOA.encode} can be used by
        L{dns.Record_SOA.decode} to reconstruct the state of the original
        L{dns.Record_SOA} instance.
        """
    self._recordRoundtripTest(dns.Record_SOA(mname=b'foo', rname=b'bar', serial=12, refresh=34, retry=56, expire=78, minimum=90))