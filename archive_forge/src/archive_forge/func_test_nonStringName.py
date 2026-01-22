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
def test_nonStringName(self):
    """
        When constructed with a name which is neither C{bytes} nor C{str},
        L{Name} raises L{TypeError}.
        """
    self.assertRaises(TypeError, dns.Name, 123)
    self.assertRaises(TypeError, dns.Name, object())
    self.assertRaises(TypeError, dns.Name, [])