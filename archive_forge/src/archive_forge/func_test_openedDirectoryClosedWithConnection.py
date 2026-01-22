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
def test_openedDirectoryClosedWithConnection(self):
    """
        A directory opened with C{openDirectory} is close when the connection
        is lost.
        """
    d = self.client.openDirectory('')
    self._emptyBuffers()

    def _getFiles(openDir):
        self.serverTransport.loseConnection()
        self.clientTransport.loseConnection()
        self.serverTransport.clearBuffer()
        self.clientTransport.clearBuffer()
        self.assertEqual(self.server.openDirs, {})
    d.addCallback(_getFiles)
    return d