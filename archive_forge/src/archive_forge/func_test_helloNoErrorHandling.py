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
def test_helloNoErrorHandling(self):
    """
        Verify that if an unknown error type is raised, it will be relayed to
        the other end of the connection and translated into an exception, it
        will be logged, and then the connection will be dropped.
        """
    L = []
    c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
    HELLO = THING_I_DONT_UNDERSTAND
    c.sendHello(HELLO).addErrback(L.append)
    p.flush()
    ure = L.pop()
    ure.trap(amp.UnknownRemoteError)
    c.sendHello(HELLO).addErrback(L.append)
    cl = L.pop()
    cl.trap(error.ConnectionDone)
    self.assertTrue(self.flushLoggedErrors(ThingIDontUnderstandError))