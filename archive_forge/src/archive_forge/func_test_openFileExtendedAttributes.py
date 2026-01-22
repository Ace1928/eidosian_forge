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
def test_openFileExtendedAttributes(self):
    """
        Check that L{filetransfer.FileTransferClient.openFile} can send
        extended attributes, that should be extracted server side. By default,
        they are ignored, so we just verify they are correctly parsed.
        """
    savedAttributes = {}
    oldOpenFile = self.server.client.openFile

    def openFile(filename, flags, attrs):
        savedAttributes.update(attrs)
        return oldOpenFile(filename, flags, attrs)
    self.server.client.openFile = openFile
    d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {'ext_foo': b'bar'})
    self._emptyBuffers()

    def check(ign):
        self.assertEqual(savedAttributes, {'ext_foo': b'bar'})
    return d.addCallback(check)