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
def test_removeFile(self):
    d = self.client.getAttrs(b'testRemoveFile')
    self._emptyBuffers()

    def _removeFile(ignored):
        d = self.client.removeFile(b'testRemoveFile')
        self._emptyBuffers()
        return d
    d.addCallback(_removeFile)
    d.addCallback(_removeFile)
    return self.assertFailure(d, filetransfer.SFTPError)