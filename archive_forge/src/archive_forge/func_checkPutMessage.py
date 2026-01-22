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
def checkPutMessage(self, transfers, randomOrder=False):
    """
        Check output of cftp client for a put request.


        @param transfers: List with tuple of (local, remote, progress).
        @param randomOrder: When set to C{True}, it will ignore the order
            in which put reposes are received

        """
    output = self.client.transport.value()
    output = output.decode('utf-8')
    output = output.split('\n\r')
    expectedOutput = []
    actualOutput = []
    for local, remote, expected in transfers:
        expectedTransfer = []
        for line in expected:
            expectedTransfer.append(f'{local} {line}')
        expectedTransfer.append(f'Transferred {local} to {remote}')
        expectedOutput.append(expectedTransfer)
        progressParts = output.pop(0).strip('\r').split('\r')
        actual = progressParts[:-1]
        last = progressParts[-1].strip('\n').split('\n')
        actual.extend(last)
        actualTransfer = []
        for line in actual[:-1]:
            line = line.strip().rsplit(' ', 2)[0]
            line = line.strip().split(' ', 1)
            actualTransfer.append(f'{line[0]} {line[1].strip()}')
        actualTransfer.append(actual[-1])
        actualOutput.append(actualTransfer)
    if randomOrder:
        self.assertEqual(sorted(expectedOutput), sorted(actualOutput))
    else:
        self.assertEqual(expectedOutput, actualOutput)
    self.assertEqual(0, len(output), 'There are still put responses which were not checked.')