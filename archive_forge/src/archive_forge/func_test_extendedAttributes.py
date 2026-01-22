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
def test_extendedAttributes(self):
    """
        Test the return of extended attributes by the server: the sftp client
        should ignore them, but still be able to parse the response correctly.

        This test is mainly here to check that
        L{filetransfer.FILEXFER_ATTR_EXTENDED} has the correct value.
        """
    env = dict(os.environ)
    fn = self.mktemp()
    with open(fn, 'w') as f:
        f.write('ls .\nexit')
    port = self.server.getHost().port
    oldGetAttr = FileTransferForTestAvatar._getAttrs

    def _getAttrs(self, s):
        attrs = oldGetAttr(self, s)
        attrs['ext_foo'] = 'bar'
        return attrs
    self.patch(FileTransferForTestAvatar, '_getAttrs', _getAttrs)
    self.server.factory.expectedLoseConnection = True
    d = getProcessValue('ssh', ('-o', 'PubkeyAcceptedKeyTypes=ssh-dss', '-V'), env)

    def hasPAKT(status):
        if status == 0:
            args = ('-o', 'PubkeyAcceptedKeyTypes=ssh-dss')
        else:
            args = ()
        args += ('-F', '/dev/null', '-o', 'IdentityFile=dsa_test', '-o', 'UserKnownHostsFile=kh_test', '-o', 'HostKeyAlgorithms=ssh-rsa', '-o', 'Port=%i' % (port,), '-b', fn, 'testuser@127.0.0.1')
        return args

    def check(result):
        self.assertEqual(result[2], 0, result[1].decode('ascii'))
        for i in [b'testDirectory', b'testRemoveFile', b'testRenameFile', b'testfile1']:
            self.assertIn(i, result[0])
    d.addCallback(hasPAKT)
    d.addCallback(lambda args: getProcessOutputAndValue('sftp', args, env))
    return d.addCallback(check)