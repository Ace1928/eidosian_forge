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
def testGet(self):
    """
        Test that 'get' saves the remote file to the correct local location,
        that the output of 'get' is correct and that 'rm' actually removes
        the file.
        """
    expectedOutput = 'Transferred {}/testfile1 to {}/test file2'.format(self.testDir.path, self.testDir.path)
    if isinstance(expectedOutput, str):
        expectedOutput = expectedOutput.encode('utf-8')

    def _checkGet(result):
        self.assertTrue(result.endswith(expectedOutput))
        self.assertFilesEqual(self.testDir.child('testfile1'), self.testDir.child('test file2'), 'get failed')
        return self.runCommand('rm "test file2"')
    d = self.runCommand(f'get testfile1 "{self.testDir.path}/test file2"')
    d.addCallback(_checkGet)
    d.addCallback(lambda _: self.assertFalse(self.testDir.child('test file2').exists()))
    return d