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
def test_TSIG(self):
    """
        The byte stream written by L{dns.Record_TSIG.encode} can be used by
        L{dns.Record_TSIG.decode} to reconstruct the state of the original
        L{dns.Record_TSIG} instance.
        """
    mac = b'\x00\x01\x02\x03\x10\x11\x12\x13 !"#0123'
    rr = dns.Record_TSIG(algorithm='hmac-md5.sig-alg.reg.int', timeSigned=1515548975, originalID=42, fudge=5, MAC=mac)
    self._recordRoundtripTest(rr)
    rdata = b'\x08hmac-md5\x07sig-alg\x03reg\x03int\x00\x00\x00ZUq/\x00\x05\x00\x10' + mac + b'\x00*\x00\x00\x00\x00'
    self.assertEncodedFormat(rdata, rr)
    rr = dns.Record_TSIG(algorithm='hmac-sha256', timeSigned=4511798055, originalID=65535, error=dns.EBADTIME, otherData=b'\x80\x00\x00\x00\x00\x08', MAC=mac)
    self._recordRoundtripTest(rr)
    rdata = b"\x0bhmac-sha256\x00\x00\x01\x0c\xec\x93'\x00\x05\x00\x10" + mac + b'\xff\xff\x00\x12\x00\x06\x80\x00\x00\x00\x00\x08'
    self.assertEncodedFormat(rdata, rr)