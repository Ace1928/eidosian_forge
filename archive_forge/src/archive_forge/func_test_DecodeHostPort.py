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
def test_DecodeHostPort(self):
    """
        Decode a host port.
        """
    self.assertEqual(ftp.decodeHostPort('25,234,129,22,100,23'), ('25.234.129.22', 25623))
    nums = range(6)
    for i in range(6):
        badValue = list(nums)
        badValue[i] = 256
        s = ','.join(map(str, badValue))
        self.assertRaises(ValueError, ftp.decodeHostPort, s)