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
def test_closedFileGetAttrs(self):
    d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {})
    self._emptyBuffers()

    def _getAttrs(_, openFile):
        d = openFile.getAttrs()
        self._emptyBuffers()
        return d

    def _err(f):
        self.flushLoggedErrors()
        return f

    def _close(openFile):
        d = openFile.close()
        self._emptyBuffers()
        d.addCallback(_getAttrs, openFile)
        d.addErrback(_err)
        return self.assertFailure(d, filetransfer.SFTPError)
    d.addCallback(_close)
    return d