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
def test_PORTCannotConnect(self):
    """
        Listen on a port, and immediately stop listening as a way to find a
        port number that is definitely closed.
        """
    d = self._anonymousLogin()

    def loggedIn(ignored):
        port = reactor.listenTCP(0, protocol.Factory(), interface='127.0.0.1')
        portNum = port.getHost().port
        d = port.stopListening()
        d.addCallback(lambda _: portNum)
        return d
    d.addCallback(loggedIn)

    def gotPortNum(portNum):
        return self.assertCommandFailed('PORT ' + ftp.encodeHostPort('127.0.0.1', portNum), ["425 Can't open data connection."])
    return d.addCallback(gotPortNum)