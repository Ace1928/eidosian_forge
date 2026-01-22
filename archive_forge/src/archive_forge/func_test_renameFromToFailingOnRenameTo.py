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
def test_renameFromToFailingOnRenameTo(self):
    """
        The L{Deferred} returned by L{ftp.FTPClient.rename} is errbacked with
        L{CommandFailed} if the I{RNTO} command receives an error response code
        (for example, because the destination directory does not exist).
        """
    self._testLogin()
    d = self.client.rename('/spam', '/ham')
    self.assertEqual(self.transport.value(), b'RNFR /spam\r\n')
    self.transport.clear()
    self.client.lineReceived(b'350 Requested file action pending further information.\r\n')
    self.assertEqual(self.transport.value(), b'RNTO /ham\r\n')
    self.client.lineReceived(b'550 Requested file unavailable.\r\n')
    return self.assertFailure(d, ftp.CommandFailed)