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
def test_helloErrorHandling(self):
    """
        Verify that if a known error type is raised and handled, it will be
        properly relayed to the other end of the connection and translated into
        an exception, and no error will be logged.
        """
    L = []
    c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
    HELLO = b'fuck you'
    c.sendHello(HELLO).addErrback(L.append)
    p.flush()
    L[0].trap(UnfriendlyGreeting)
    self.assertEqual(str(L[0].value), "Don't be a dick.")