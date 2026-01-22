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
def test_LISTUnicode(self):
    """
        Unicode filenames returned from L{IFTPShell.list} are encoded using
        UTF-8 before being sent with the response.
        """
    return self._listTestHelper('LIST', ('my resumé', (0, 1, filepath.Permissions(511), 0, 0, 'user', 'group')), b'drwxrwxrwx   0 user      group                   0 Jan 01  1970 my resum\xc3\xa9\r\n')