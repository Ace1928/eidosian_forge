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
def test_cmd_PUTSingleRemotePath(self):
    """
        Remote path is extracted from first filename after local file.

        Any other data in the line is ignored.
        """
    localPath = self.makeFile()
    flags = filetransfer.FXF_WRITE | filetransfer.FXF_CREAT | filetransfer.FXF_TRUNC
    remoteName = '/remote-path'
    remoteFile = InMemoryRemoteFile(remoteName)
    self.fakeFilesystem.put(remoteName, flags, defer.succeed(remoteFile))
    deferred = self.client.cmd_PUT(f'{localPath} {remoteName} ignored')
    self.successResultOf(deferred)
    self.checkPutMessage([(localPath, remoteName, ['100% 0.0B'])])
    self.assertTrue(remoteFile._closed)
    self.assertEqual(b'', remoteFile.getvalue())