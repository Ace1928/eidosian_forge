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
def test_printProgressBarNoProgress(self):
    """
        L{StdioClient._printProgressBar} prints a progress description that
        indicates 0 bytes transferred if no bytes have been transferred and no
        time has passed.
        """
    self.setKnownConsoleSize(10, 34)
    clock = self.client.reactor = Clock()
    wrapped = BytesIO(b'x')
    wrapped.name = b'sample'
    wrapper = cftp.FileWrapper(wrapped)
    startTime = clock.seconds()
    self.client._printProgressBar(wrapper, startTime)
    result = b"\rb'sample'  0% 0.0B 0.0Bps 00:00 "
    self.assertEqual(self.client.transport.value(), result)