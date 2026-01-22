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
def test_sendBox(self):
    """
        When a binary box protocol sends a box, it should emit the serialized
        bytes of that box to its transport.
        """
    a = amp.BinaryBoxProtocol(self)
    a.makeConnection(self)
    aBox = amp.Box({b'testKey': b'valueTest', b'someData': b'hello'})
    a.makeConnection(self)
    a.sendBox(aBox)
    self.assertEqual(b''.join(self.data), aBox.serialize())