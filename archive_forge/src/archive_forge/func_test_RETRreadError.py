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
def test_RETRreadError(self):
    """
        Any errors during reading a file inside a RETR should be returned to
        the client.
        """

    class FailingFileReader(ftp._FileReader):

        def send(self, consumer):
            return defer.fail(ftp.IsADirectoryError('blah'))

    def failingRETR(a, b):
        return defer.succeed(FailingFileReader(None))
    self.patch(ftp.FTPAnonymousShell, 'openForReading', failingRETR)

    def check_response(failure):
        self.flushLoggedErrors()
        failure.trap(ftp.CommandFailed)
        self.assertEqual(failure.value.args[0][0], '125 Data connection already open, starting transfer')
        self.assertEqual(failure.value.args[0][1], '550 blah: is a directory')
    proto = _BufferingProtocol()
    d = self.client.retrieveFile('failing_file', proto)
    d.addErrback(check_response)
    return d