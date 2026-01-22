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
def test_errbacksUponDisconnect(self):
    """
        Test the ftp command errbacks when a connection lost happens during
        the operation.
        """
    ftpClient = ftp.FTPClient()
    tr = proto_helpers.StringTransportWithDisconnection()
    ftpClient.makeConnection(tr)
    tr.protocol = ftpClient
    d = ftpClient.list('some path', Dummy())
    m = []

    def _eb(failure):
        m.append(failure)
        return None
    d.addErrback(_eb)
    from twisted.internet.main import CONNECTION_LOST
    ftpClient.connectionLost(failure.Failure(CONNECTION_LOST))
    self.assertTrue(m, m)
    return d