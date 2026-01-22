from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
def test_serverGreeting(self, write='write'):
    """
        Test that on a connection where the server speaks first, the client
        receives the bytes sent by the server.
        """
    return self._greetingtest('write', True)