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
def test_lateAnswer(self):
    """
        Verify that a command that does not get answered until after the
        connection terminates will not cause any errors.
        """
    c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
    L = []
    c.callRemote(WaitForever).addErrback(L.append)
    p.flush()
    self.assertEqual(L, [])
    s.transport.loseConnection()
    p.flush()
    L.pop().trap(error.ConnectionDone)
    s.waiting.callback({})
    return s.waiting