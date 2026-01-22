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
def test_linkPath(self):
    d = self.client.makeLink(b'testLink', b'testfile1')
    self._emptyBuffers()

    def _readLink(_):
        d = self.client.readLink(b'testLink')
        self._emptyBuffers()
        testFile = FilePath(os.getcwd()).preauthChild(self.testDir.path)
        testFile = testFile.child('testfile1')
        d.addCallback(self.assertEqual, testFile.path)
        return d

    def _realPath(_):
        d = self.client.realPath(b'testLink')
        self._emptyBuffers()
        testLink = FilePath(os.getcwd()).preauthChild(self.testDir.path)
        testLink = testLink.child('testfile1')
        d.addCallback(self.assertEqual, testLink.path)
        return d
    d.addCallback(_readLink)
    d.addCallback(_realPath)
    return d