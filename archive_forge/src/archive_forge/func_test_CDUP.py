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
def test_CDUP(self):
    """
        Test the CDUP command.

        L{ftp.FTPClient.cdup} should return a Deferred which fires with a
        sequence of one element which is the string the server sent
        indicating that the command was executed successfully.

        (XXX - This is a bad API)
        """

    def cbCdup(res):
        self.assertEqual(res[0], '250 Requested File Action Completed OK')
    self._testLogin()
    d = self.client.cdup().addCallback(cbCdup)
    self.assertEqual(self.transport.value(), b'CDUP\r\n')
    self.transport.clear()
    self.client.lineReceived(b'250 Requested File Action Completed OK')
    return d