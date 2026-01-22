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
def test_connectionAfterTimeout(self):
    """
        If L{ftp.DTPFactory.buildProtocol} is called after the timeout
        specified by L{ftp.DTPFactory.setTimeout} has elapsed, L{None} is
        returned.
        """
    d = self.assertFailure(self.factory.deferred, ftp.PortConnectionError)
    self.factory.setTimeout(10)
    self.reactor.advance(10)
    self.assertIsNone(self.factory.buildProtocol(None))
    return d