import getpass
import locale
import operator
import os
import struct
import sys
import time
from io import BytesIO, TextIOWrapper
from unittest import skipIf
from zope.interface import implementer
from twisted.conch import ls
from twisted.conch.interfaces import ISFTPFile
from twisted.conch.test.test_filetransfer import FileTransferTestAvatar, SFTPTestBase
from twisted.cred import portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.internet.utils import getProcessOutputAndValue, getProcessValue
from twisted.python import log
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def testChAttrs(self):
    """
        Check that 'ls -l' output includes the access permissions and that
        this output changes appropriately with 'chmod'.
        """

    def _check(results):
        self.flushLoggedErrors()
        self.assertTrue(results[0].startswith(b'-rw-r--r--'))
        self.assertEqual(results[1], b'')
        self.assertTrue(results[2].startswith(b'----------'), results[2])
        self.assertEqual(results[3], b'')
    d = self.runScript('ls -l testfile1', 'chmod 0 testfile1', 'ls -l testfile1', 'chmod 644 testfile1')
    return d.addCallback(_check)