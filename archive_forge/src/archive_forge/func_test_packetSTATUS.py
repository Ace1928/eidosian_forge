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
def test_packetSTATUS(self):
    """
        A STATUS packet containing a result code, a message, and a language is
        parsed to produce the result of an outstanding request L{Deferred}.

        @see: U{section 9.1<http://tools.ietf.org/html/draft-ietf-secsh-filexfer-13#section-9.1>}
            of the SFTP Internet-Draft.
        """
    d = defer.Deferred()
    d.addCallback(self._cbTestPacketSTATUS)
    self.ftc.openRequests[1] = d
    data = struct.pack('!LL', 1, filetransfer.FX_OK) + common.NS(b'msg') + common.NS(b'lang')
    self.ftc.packet_STATUS(data)
    return d