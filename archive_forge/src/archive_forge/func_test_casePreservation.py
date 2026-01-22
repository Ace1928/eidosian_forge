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
def test_casePreservation(self):
    """
        L{dns._nameToLabels} preserves the case of ascii
        characters in labels.
        """
    self.assertEqual(dns._nameToLabels(b'EXAMPLE.COM'), [b'EXAMPLE', b'COM', b''])