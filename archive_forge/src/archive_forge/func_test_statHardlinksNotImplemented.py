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
def test_statHardlinksNotImplemented(self):
    """
        If L{twisted.python.filepath.FilePath.getNumberOfHardLinks} is not
        implemented, the number returned is 0
        """
    pathFunc = self.shell._path

    def raiseNotImplemented():
        raise NotImplementedError

    def notImplementedFilePath(path):
        f = pathFunc(path)
        f.getNumberOfHardLinks = raiseNotImplemented
        return f
    self.shell._path = notImplementedFilePath
    self.createDirectory('ned')
    d = self.shell.stat(('ned',), ('hardlinks',))
    self.assertEqual(self.successResultOf(d), [0])