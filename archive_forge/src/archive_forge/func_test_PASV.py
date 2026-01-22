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
def test_PASV(self):
    """
        When the client sends the command C{PASV}, the server responds with a
        host and port, and is listening on that port.
        """
    d = self._anonymousLogin()
    d.addCallback(lambda _: self.client.queueStringCommand('PASV'))

    def cb(responseLines):
        """
            Extract the host and port from the resonse, and
            verify the server is listening of the port it claims to be.
            """
        host, port = ftp.decodeHostPort(responseLines[-1][4:])
        self.assertEqual(port, self.serverProtocol.dtpPort.getHost().port)
    d.addCallback(cb)
    d.addCallback(lambda _: self.serverProtocol.transport.loseConnection())
    return d