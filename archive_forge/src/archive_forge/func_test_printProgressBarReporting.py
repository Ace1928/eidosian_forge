import getpass
import locale
import operator
import os
import struct
import sys
import time
from io import BytesIO, TextIOWrapper
from unittest import skipIf
from zope.interface import implementer
from twisted.conch import ls
from twisted.conch.interfaces import ISFTPFile
from twisted.conch.test.test_filetransfer import FileTransferTestAvatar, SFTPTestBase
from twisted.cred import portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.internet.utils import getProcessOutputAndValue, getProcessValue
from twisted.python import log
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_printProgressBarReporting(self):
    """
        L{StdioClient._printProgressBar} prints a progress description,
        including percent done, amount transferred, transfer rate, and time
        remaining, all based the given start time, the given L{FileWrapper}'s
        progress information and the reactor's current time.
        """
    self.setKnownConsoleSize(10, 34)
    clock = self.client.reactor = Clock()
    wrapped = BytesIO(b'x')
    wrapped.name = b'sample'
    wrapper = cftp.FileWrapper(wrapped)
    wrapper.size = 1024 * 10
    startTime = clock.seconds()
    clock.advance(2.0)
    wrapper.total += 4096
    self.client._printProgressBar(wrapper, startTime)
    result = b"\rb'sample' 40% 4.0kB 2.0kBps 00:03 "
    self.assertEqual(self.client.transport.value(), result)