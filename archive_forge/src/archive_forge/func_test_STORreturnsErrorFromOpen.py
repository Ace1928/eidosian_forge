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
def test_STORreturnsErrorFromOpen(self):
    """
        Any FTP error raised inside STOR while opening the file is returned
        to the client.
        """
    self.dirPath.child(self.username).createDirectory()
    self.dirPath.child(self.username).child('folder').createDirectory()
    d = self._userLogin()

    def sendPASV(result):
        """
            Send the PASV command required before port.
            """
        return self.client.queueStringCommand('PASV')

    def mockDTPInstance(result):
        """
            Fake an incoming connection and create a mock DTPInstance so
            that PORT command will start processing the request.
            """
        self.serverProtocol.dtpFactory.deferred.callback(None)
        self.serverProtocol.dtpInstance = object()
        return result
    d.addCallback(sendPASV)
    d.addCallback(mockDTPInstance)
    self.assertCommandFailed('STOR folder', ['550 folder: is a directory'], chainDeferred=d)
    return d