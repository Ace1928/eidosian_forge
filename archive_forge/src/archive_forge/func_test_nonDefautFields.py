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
def test_nonDefautFields(self):
    """
        L{dns._compactRepr} displays field values if they differ from their
        defaults.
        """
    m = self.messageFactory(field1=10, field2=20)
    self.assertEqual("<Foo field1=10 field2=20 alwaysShowField='AS' flags=flagTrue>", repr(m))