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
def test_listWithStat(self):
    """
        Check the output of list with asked stats.
        """
    self.createDirectory('ned')
    self.createFile('file.txt')
    d = self.shell.list(('.',), ('size', 'permissions'))

    def cb(l):
        l.sort()
        self.assertEqual(len(l), 2)
        self.assertEqual(l[0][0], 'file.txt')
        self.assertEqual(l[1][0], 'ned')
        self.assertEqual(len(l[0][1]), 2)
        self.assertEqual(len(l[1][1]), 2)
    return d.addCallback(cb)