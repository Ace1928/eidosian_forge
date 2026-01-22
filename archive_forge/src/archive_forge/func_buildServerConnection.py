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
def buildServerConnection(self):
    conn = connection.SSHConnection()

    class DummyTransport:

        def __init__(self):
            self.transport = self

        def sendPacket(self, kind, data):
            pass

        def logPrefix(self):
            return 'dummy transport'
    conn.transport = DummyTransport()
    conn.transport.avatar = self.avatar
    return conn