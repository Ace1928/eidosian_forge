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
def test_renameFromTo(self):
    """
        L{ftp.FTPClient.rename} issues I{RNTO} and I{RNFR} commands and returns
        a L{Deferred} which fires when a file has successfully been renamed.
        """
    self._testLogin()
    d = self.client.rename('/spam', '/ham')
    self.assertEqual(self.transport.value(), b'RNFR /spam\r\n')
    self.transport.clear()
    fromResponse = '350 Requested file action pending further information.\r\n'
    self.client.lineReceived(fromResponse.encode(self.client._encoding))
    self.assertEqual(self.transport.value(), b'RNTO /ham\r\n')
    toResponse = '250 Requested File Action Completed OK'
    self.client.lineReceived(toResponse.encode(self.client._encoding))
    d.addCallback(self.assertEqual, ([fromResponse], [toResponse]))
    return d