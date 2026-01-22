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
def test_cmd_PUTMultipleWithRemotePath(self):
    """
        When a gobbing expression is used local files are transferred with
        remote file names based on local names.
        when a remote folder is requested remote paths are composed from
        remote path and local filename.
        """
    first = self.makeFile()
    firstName = os.path.basename(first)
    secondName = 'second-name'
    parent = os.path.dirname(first)
    second = self.makeFile(path=os.path.join(parent, secondName))
    flags = filetransfer.FXF_WRITE | filetransfer.FXF_CREAT | filetransfer.FXF_TRUNC
    firstRemoteFile = InMemoryRemoteFile(firstName)
    secondRemoteFile = InMemoryRemoteFile(secondName)
    firstRemotePath = f'/remote/{firstName}'
    secondRemotePath = f'/remote/{secondName}'
    self.fakeFilesystem.put(firstRemotePath, flags, defer.succeed(firstRemoteFile))
    self.fakeFilesystem.put(secondRemotePath, flags, defer.succeed(secondRemoteFile))
    deferred = self.client.cmd_PUT('{} remote'.format(os.path.join(parent, '*')))
    self.successResultOf(deferred)
    self.assertTrue(firstRemoteFile._closed)
    self.assertEqual(b'', firstRemoteFile.getvalue())
    self.assertTrue(secondRemoteFile._closed)
    self.assertEqual(b'', secondRemoteFile.getvalue())
    self.checkPutMessage([(first, firstName, ['100% 0.0B']), (second, secondName, ['100% 0.0B'])], randomOrder=True)