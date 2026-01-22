from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_cacheDefault(self):
    """
        L{server.DNSServerFactory.cache} is L{None} by default.
        """
    self.assertIsNone(server.DNSServerFactory().cache)