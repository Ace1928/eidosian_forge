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
def test_passiveLIST(self):
    """
        Test the LIST command.

        L{ftp.FTPClient.list} should return a Deferred which fires with a
        protocol instance which was passed to list after the command has
        succeeded.

        (XXX - This is a very unfortunate API; if my understanding is
        correct, the results are always at least line-oriented, so allowing
        a per-line parser function to be specified would make this simpler,
        but a default implementation should really be provided which knows
        how to deal with all the formats used in real servers, so
        application developers never have to care about this insanity.  It
        would also be nice to either get back a Deferred of a list of
        filenames or to be able to consume the files as they are received
        (which the current API does allow, but in a somewhat inconvenient
        fashion) -exarkun)
        """

    def cbList(res, fileList):
        fls = [f['filename'] for f in fileList.files]
        expected = ['foo', 'bar', 'baz']
        expected.sort()
        fls.sort()
        self.assertEqual(fls, expected)

    def cbConnect(host, port, factory):
        self.assertEqual(host, '127.0.0.1')
        self.assertEqual(port, 12345)
        proto = factory.buildProtocol((host, port))
        proto.makeConnection(proto_helpers.StringTransport())
        self.client.lineReceived(b'150 File status okay; about to open data connection.')
        sending = [b'-rw-r--r--    0 spam      egg      100 Oct 10 2006 foo\r\n', b'-rw-r--r--    3 spam      egg      100 Oct 10 2006 bar\r\n', b'-rw-r--r--    4 spam      egg      100 Oct 10 2006 baz\r\n']
        for i in sending:
            proto.dataReceived(i)
        proto.connectionLost(failure.Failure(error.ConnectionDone('')))
    self.client.connectFactory = cbConnect
    self._testLogin()
    fileList = ftp.FTPFileListProtocol()
    d = self.client.list('foo/bar', fileList).addCallback(cbList, fileList)
    self.assertEqual(self.transport.value(), b'PASV\r\n')
    self.transport.clear()
    self.client.lineReceived(passivemode_msg(self.client))
    self.assertEqual(self.transport.value(), b'LIST foo/bar\r\n')
    self.client.lineReceived(b'226 Transfer Complete.')
    return d