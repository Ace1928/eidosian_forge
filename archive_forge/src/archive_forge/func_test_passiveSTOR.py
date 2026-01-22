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
def test_passiveSTOR(self):
    """
        Test the STOR command: send a file and verify its content.

        L{ftp.FTPClient.storeFile} should return a two-tuple of Deferreds.
        The first of which should fire with a protocol instance when the
        data connection has been established and is responsible for sending
        the contents of the file.  The second of which should fire when the
        upload has completed, the data connection has been closed, and the
        server has acknowledged receipt of the file.

        (XXX - storeFile should take a producer as an argument, instead, and
        only return a Deferred which fires when the upload has succeeded or
        failed).
        """
    tr = proto_helpers.StringTransport()

    def cbStore(sender):
        self.client.lineReceived(b'150 File status okay; about to open data connection.')
        sender.transport.write(b'x' * 1000)
        sender.finish()
        sender.connectionLost(failure.Failure(error.ConnectionDone('')))

    def cbFinish(ign):
        self.assertEqual(tr.value(), b'x' * 1000)

    def cbConnect(host, port, factory):
        self.assertEqual(host, '127.0.0.1')
        self.assertEqual(port, 12345)
        proto = factory.buildProtocol((host, port))
        proto.makeConnection(tr)
    self.client.connectFactory = cbConnect
    self._testLogin()
    d1, d2 = self.client.storeFile('spam')
    d1.addCallback(cbStore)
    d2.addCallback(cbFinish)
    self.assertEqual(self.transport.value(), b'PASV\r\n')
    self.transport.clear()
    self.client.lineReceived(passivemode_msg(self.client))
    self.assertEqual(self.transport.value(), b'STOR spam\r\n')
    self.transport.clear()
    self.client.lineReceived(b'226 Transfer Complete.')
    return defer.gatherResults([d1, d2])