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
def test_RNFRwithoutRNTO(self):
    """
        Sending the RNFR command followed by any command other than RNTO
        should return an error informing users that RNFR should be followed
        by RNTO.
        """
    d = self._anonymousLogin()
    self.assertCommandResponse('RNFR foo', ['350 Requested file action pending further information.'], chainDeferred=d)
    self.assertCommandFailed('OTHER don-tcare', ['503 Incorrect sequence of commands: RNTO required after RNFR'], chainDeferred=d)
    return d