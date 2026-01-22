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
def test_ord2byte(self):
    """
        L{dns._ord2byte} accepts an integer and returns a byte string of length
        one with an ordinal value equal to the given integer.
        """
    self.assertEqual(b'\x10', dns._ord2bytes(16))