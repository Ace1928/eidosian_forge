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
def cbStore(sender):
    self.assertEqual(self.transport.value(), 'PORT {}\r\n'.format(ftp.encodeHostPort('127.0.0.1', 9876)).encode(self.client._encoding))
    self.transport.clear()
    self.client.lineReceived(b'200 PORT OK')
    self.assertEqual(self.transport.value(), b'STOR spam\r\n')
    self.transport.clear()
    self.client.lineReceived(b'150 File status okay; about to open data connection.')
    sender.transport.write(b'x' * 1000)
    sender.finish()
    sender.connectionLost(failure.Failure(error.ConnectionDone('')))
    self.client.lineReceived(b'226 Transfer Complete.')