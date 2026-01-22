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
def test_requiresNoAnswerAfterFail(self):
    """
        No-answer commands sent after the connection has been torn down do not
        return a L{Deferred}.
        """
    c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
    c.transport.loseConnection()
    p.flush()
    result = c.callRemote(NoAnswerHello, hello=b'ignored')
    self.assertIs(result, None)