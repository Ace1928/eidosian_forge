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
def test_openFileIO(self):
    d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {})
    self._emptyBuffers()

    def _fileOpened(openFile):
        self.assertEqual(openFile, filetransfer.ISFTPFile(openFile))
        d = _readChunk(openFile)
        d.addCallback(_writeChunk, openFile)
        return d

    def _readChunk(openFile):
        d = openFile.readChunk(0, 20)
        self._emptyBuffers()
        d.addCallback(self.assertEqual, b'a' * 10 + b'b' * 10)
        return d

    def _writeChunk(_, openFile):
        d = openFile.writeChunk(20, b'c' * 10)
        self._emptyBuffers()
        d.addCallback(_readChunk2, openFile)
        return d

    def _readChunk2(_, openFile):
        d = openFile.readChunk(0, 30)
        self._emptyBuffers()
        d.addCallback(self.assertEqual, b'a' * 10 + b'b' * 10 + b'c' * 10)
        return d
    d.addCallback(_fileOpened)
    return d