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
def test_helloWorldUnicode(self):
    """
        Verify that unicode arguments can be encoded and decoded.
        """
    c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
    L = []
    HELLO = b'world'
    HELLO_UNICODE = 'worሴld'
    c.sendUnicodeHello(HELLO, HELLO_UNICODE).addCallback(L.append)
    p.flush()
    self.assertEqual(L[0]['hello'], HELLO)
    self.assertEqual(L[0]['Print'], HELLO_UNICODE)