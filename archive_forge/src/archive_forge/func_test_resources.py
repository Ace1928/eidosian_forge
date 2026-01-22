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
def test_resources(self):
    """
        L{dns.SimpleRecord.encode} encodes the record's name information and
        writes it to the file-like object passed to it and
        L{dns.SimpleRecord.decode} reads from a file-like object to re-construct
        a L{dns.SimpleRecord} instance.
        """
    names = (b'this.are.test.name', b'will.compress.will.this.will.name.will.hopefully', b'test.CASE.preSErVatIOn.YeAH', b'a.s.h.o.r.t.c.a.s.e.t.o.t.e.s.t', b'singleton')
    for s in names:
        f = BytesIO()
        dns.SimpleRecord(s).encode(f)
        f.seek(0, 0)
        result = dns.SimpleRecord()
        result.decode(f)
        self.assertEqual(result.name, dns.Name(s))