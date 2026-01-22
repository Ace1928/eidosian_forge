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
def test_stopConnectionServiceClosesChannel(self):
    """
        Closing an SSH connection should close all sessions within it.
        """
    conn = self.buildServerConnection()
    packet = common.NS(b'session') + struct.pack('>L', 0) * 3
    conn.ssh_CHANNEL_OPEN(packet)
    sessionChannel = conn.channels[0]
    sessionChannel.request_subsystem(common.NS(b'sftp'))
    sftpServer = sessionChannel.client.transport.proto
    self.interceptConnectionLost(sftpServer)
    conn.serviceStopped()
    self.assertSFTPConnectionLost()