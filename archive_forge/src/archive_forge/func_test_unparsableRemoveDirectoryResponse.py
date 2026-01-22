import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
def test_unparsableRemoveDirectoryResponse(self):
    """
        If the server returns a response line which cannot be parsed, the
        L{Deferred} returned by L{ftp.FTPClient.removeDirectory} is errbacked
        with a L{BadResponse} containing the response.
        """
    self._testLogin()
    d = self.client.removeDirectory('/tmp/test')
    response = '765 blah blah blah'
    self.client.lineReceived(response.encode(self.client._encoding))
    d = self.assertFailure(d, ftp.BadResponse)
    d.addCallback(lambda exc: self.assertEqual(exc.args, ([response],)))
    return d