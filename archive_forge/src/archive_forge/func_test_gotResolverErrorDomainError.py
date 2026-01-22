from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_gotResolverErrorDomainError(self):
    """
        L{server.DNSServerFactory.gotResolver} triggers a response message with
        an C{rCode} of L{dns.ENAME} if supplied with a L{error.DomainError}.
        """
    self._assertMessageRcodeForError(error.DomainError(), dns.ENAME)