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
def test_noSuchUser(self):
    """
        L{ftp.SystemFTPRealm.getHomeDirectory} raises L{UnauthorizedLogin} when
        passed a username which has no corresponding home directory in the
        system's accounts database.
        """
    user = random.choice(string.ascii_letters) + ''.join((random.choice(string.ascii_letters + string.digits) for _ in range(4)))
    realm = ftp.SystemFTPRealm(self.mktemp())
    self.assertRaises(UnauthorizedLogin, realm.getHomeDirectory, user)