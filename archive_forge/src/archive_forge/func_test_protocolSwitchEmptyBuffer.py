import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
def test_protocolSwitchEmptyBuffer(self):
    """
        After switching to a different protocol, if no extra bytes beyond
        the switch box were delivered, an empty string is not passed to the
        switched protocol's C{dataReceived} method.
        """
    a = amp.BinaryBoxProtocol(self)
    a.makeConnection(self)
    otherProto = TestProto(None, b'')
    a._switchTo(otherProto)
    self.assertEqual(otherProto.data, [])