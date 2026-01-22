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
def test_optionalAmpListOmitted(self):
    """
        Sending a command with an omitted AmpList argument that is
        designated as optional does not raise an InvalidSignature error.
        """
    c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
    L = []
    c.callRemote(DontRejectMe, magicWord='please').addCallback(L.append)
    p.flush()
    response = L.pop().get('response')
    self.assertEqual(response, 'list omitted')