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
def test_PWD(self):
    """
        Test the PWD command.

        L{ftp.FTPClient.pwd} should return a Deferred which fires with a
        sequence of one element which is a string representing the current
        working directory on the server.

        (XXX - This is a bad API)
        """

    def cbPwd(res):
        self.assertEqual(ftp.parsePWDResponse(res[0]), '/bar/baz')
    self._testLogin()
    d = self.client.pwd().addCallback(cbPwd)
    self.assertEqual(self.transport.value(), b'PWD\r\n')
    self.client.lineReceived(b'257 "/bar/baz"')
    return d