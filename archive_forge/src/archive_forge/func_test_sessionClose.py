import os
import re
import struct
from unittest import skipIf
from hamcrest import assert_that, equal_to
from twisted.internet import defer
from twisted.internet.error import ConnectionLost
from twisted.internet.testing import StringTransport
from twisted.protocols import loopback
from twisted.python import components
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
def test_sessionClose(self):
    """
        Closing a session should notify an SFTP subsystem launched by that
        session.
        """
    testSession = session.SSHSession(conn=FakeConn(), avatar=self.avatar)
    testSession.request_subsystem(common.NS(b'sftp'))
    sftpServer = testSession.client.transport.proto
    self.interceptConnectionLost(sftpServer)
    testSession.closeReceived()
    self.assertSFTPConnectionLost()