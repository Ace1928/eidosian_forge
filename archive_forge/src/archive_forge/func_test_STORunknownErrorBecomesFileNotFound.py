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
def test_STORunknownErrorBecomesFileNotFound(self):
    """
        Any non FTP error raised inside STOR while opening the file is
        converted into FileNotFound error and returned to the client together
        with the path.

        The unknown error is logged.
        """
    d = self._userLogin()

    def failingOpenForWriting(ignore):
        """
            Override openForWriting.

            @param ignore: ignored, used for callback
            @return: an error
            """
        return defer.fail(AssertionError())

    def sendPASV(result):
        """
            Send the PASV command required before port.

            @param result: parameter used in L{Deferred}
            """
        return self.client.queueStringCommand('PASV')

    def mockDTPInstance(result):
        """
            Fake an incoming connection and create a mock DTPInstance so
            that PORT command will start processing the request.

            @param result: parameter used in L{Deferred}
            """
        self.serverProtocol.dtpFactory.deferred.callback(None)
        self.serverProtocol.dtpInstance = object()
        self.serverProtocol.shell.openForWriting = failingOpenForWriting
        return result

    def checkLogs(result):
        """
            Check that unknown errors are logged.

            @param result: parameter used in L{Deferred}
            """
        logs = self.flushLoggedErrors()
        self.assertEqual(1, len(logs))
        self.assertIsInstance(logs[0].value, AssertionError)
    d.addCallback(sendPASV)
    d.addCallback(mockDTPInstance)
    self.assertCommandFailed('STOR something', ['550 something: No such file or directory.'], chainDeferred=d)
    d.addCallback(checkLogs)
    return d