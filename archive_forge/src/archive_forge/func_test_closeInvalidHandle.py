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
def test_closeInvalidHandle(self):
    """
        A close request with an unknown handle receives an FX_NO_SUCH_FILE error
        response.
        """
    transport = StringTransport()
    self.fts.makeConnection(transport)
    requestId = b'1234'
    handle = b'invalid handle'
    close = common.NS(bytes([4]) + requestId + common.NS(handle))
    self.fts.dataReceived(close)
    expected = common.NS(bytes([101]) + requestId + bytes([0, 0, 0, 2]) + common.NS(b'No such file or directory') + common.NS(b''))
    assert_that(transport.value(), equal_to(expected))