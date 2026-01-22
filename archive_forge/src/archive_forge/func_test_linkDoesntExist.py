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
def test_linkDoesntExist(self):
    d = self.client.getAttrs(b'testLink')
    self._emptyBuffers()
    return self.assertFailure(d, filetransfer.SFTPError)