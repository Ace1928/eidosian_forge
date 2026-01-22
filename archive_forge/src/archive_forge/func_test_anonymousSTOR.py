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
def test_anonymousSTOR(self):
    """
        Try to make an STOR as anonymous, and check that we got a permission
        denied error.
        """

    def eb(res):
        res.trap(ftp.CommandFailed)
        self.assertEqual(res.value.args[0][0], '550 foo: Permission denied.')
    d1, d2 = self.client.storeFile('foo')
    d2.addErrback(eb)
    return defer.gatherResults([d1, d2])