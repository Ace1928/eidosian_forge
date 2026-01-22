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
def test_renameFromToFailingOnFirstError(self):
    """
        The L{Deferred} returned by L{ftp.FTPClient.rename} is errbacked with
        L{CommandFailed} if the I{RNFR} command receives an error response code
        (for example, because the file does not exist).
        """
    self._testLogin()
    d = self.client.rename('/spam', '/ham')
    self.assertEqual(self.transport.value(), b'RNFR /spam\r\n')
    self.transport.clear()
    self.client.lineReceived(b'550 Requested file unavailable.\r\n')
    self.assertEqual(self.transport.value(), b'')
    return self.assertFailure(d, ftp.CommandFailed)