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
def test_Year(self):
    """
        This example derived from bug description in issue 514.

        @return: L{Deferred} of command response
        """
    fileList = ftp.FTPFileListProtocol()
    exampleLine = b'-rw-r--r--   1 root     other        531 Jan 29 2003 README\n'

    class PrintLine(protocol.Protocol):

        def connectionMade(self):
            self.transport.write(exampleLine)
            self.transport.loseConnection()

    def check(ignored):
        file = fileList.files[0]
        self.assertTrue(file['size'] == 531, 'misparsed fileitem')
        self.assertTrue(file['date'] == 'Jan 29 2003', 'misparsed fileitem')
        self.assertTrue(file['filename'] == 'README', 'misparsed fileitem')
    d = loopback.loopbackAsync(PrintLine(), fileList)
    return d.addCallback(check)