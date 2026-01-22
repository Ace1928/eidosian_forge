from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_gotResolverErrorOtherError(self):
    """
        L{server.DNSServerFactory.gotResolver} triggers a response message with
        an C{rCode} of L{dns.ESERVER} if supplied with another type of error and
        logs the error.
        """
    self._assertMessageRcodeForError(KeyError(), dns.ESERVER)
    e = self.flushLoggedErrors(KeyError)
    self.assertEqual(len(e), 1)