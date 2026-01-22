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
def test_openFileAttributes(self):
    d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {})
    self._emptyBuffers()

    def _getAttrs(openFile):
        d = openFile.getAttrs()
        self._emptyBuffers()
        d.addCallback(_getAttrs2)
        return d

    def _getAttrs2(attrs1):
        d = self.client.getAttrs(b'testfile1')
        self._emptyBuffers()
        d.addCallback(self.assertEqual, attrs1)
        return d
    return d.addCallback(_getAttrs)