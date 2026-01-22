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
def test_isGlobbingExpressionGlob(self):
    """
        _isGlobbingExpression will return True for segments which contains
        globbing characters in the last segment part (filename).
        """
    self.assertTrue(ftp._isGlobbingExpression(['ignore', '*.txt']))
    self.assertTrue(ftp._isGlobbingExpression(['ignore', '[a-b].txt']))
    self.assertTrue(ftp._isGlobbingExpression(['ignore', 'fil?.txt']))