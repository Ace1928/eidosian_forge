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
def test_LISTNonASCIIBytes(self):
    """
        When LIST receive a filename as byte string from L{IFTPShell.list}
        it will just pass the data to lower level without any change.

        @return: L{_listTestHelper}
        """
    return self._listTestHelper('LIST', (b'my resum\xc3\xa9', (0, 1, filepath.Permissions(511), 0, 0, 'user', 'group')), b'drwxrwxrwx   0 user      group                   0 Jan 01  1970 my resum\xc3\xa9\r\n')