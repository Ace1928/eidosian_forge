from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
def test_clientHost(self, get='getHost'):
    """
        Test that the client gets a transport with a properly functioning
        implementation of L{ITransport.getHost}.
        """
    return self._hostpeertest('getHost', False)