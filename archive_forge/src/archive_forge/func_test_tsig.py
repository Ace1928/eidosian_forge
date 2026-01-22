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
def test_tsig(self):
    """
        L{dns.Record_TSIG} instances compare equal if and only if they have the
        same RDATA (algorithm, timestamp, MAC, etc.) and ttl.
        """
    baseargs = {'algorithm': 'hmac-sha224', 'timeSigned': 1515548975, 'fudge': 5, 'MAC': b'\x01\x02\x03\x04\x05', 'originalID': 99, 'error': dns.OK, 'otherData': b'', 'ttl': 40}
    altargs = {'algorithm': 'hmac-sha512', 'timeSigned': 1515548875, 'fudge': 0, 'MAC': b'\x05\x04\x03\x02\x01', 'originalID': 65437, 'error': dns.EBADTIME, 'otherData': b'\x00\x00', 'ttl': 400}
    for kw in baseargs.keys():
        altered = baseargs.copy()
        altered[kw] = altargs[kw]
        self._equalityTest(dns.Record_TSIG(**altered), dns.Record_TSIG(**altered), dns.Record_TSIG(**baseargs))