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
def test_noAnswerResponderAskedForAnswer(self):
    """
        Verify that responders with requiresAnswer=False will actually respond
        if the client sets requiresAnswer=True.  In other words, verify that
        requiresAnswer is a hint honored only by the client.
        """
    c, s, p = connectedServerAndClient(ServerClass=NoAnswerCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
    L = []
    c.callRemote(Hello, hello=b'Hello!').addCallback(L.append)
    p.flush()
    self.assertEqual(len(L), 1)
    self.assertEqual(L, [dict(hello=b'Hello!-noanswer', Print=None)])