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
def test_receiveLongerBoxData(self):
    """
        An L{amp.BinaryBoxProtocol} can receive serialized AMP boxes with
        values of up to (2 ** 16 - 1) bytes.
        """
    length = 2 ** 16 - 1
    value = b'x' * length
    transport = StringTransport()
    protocol = amp.BinaryBoxProtocol(self)
    protocol.makeConnection(transport)
    protocol.dataReceived(amp.Box({'k': value}).serialize())
    self.assertEqual(self.boxes, [amp.Box({'k': value})])
    self.assertFalse(transport.disconnecting)