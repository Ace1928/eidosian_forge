import os
from binascii import Error as BinasciiError, a2b_base64, b2a_base64
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.internet.defer import Deferred
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial.unittest import TestCase
def test_promptYes(self):
    """
        L{ConsoleUI.prompt} writes a message to the console, then reads a line.
        If that line is 'yes', then it returns a L{Deferred} that fires with
        True.
        """
    for okYes in [b'yes', b'Yes', b'yes\n']:
        self.newFile([okYes])
        l = []
        self.ui.prompt('Hello, world!').addCallback(l.append)
        self.assertEqual(['Hello, world!'], self.fakeFile.outchunks)
        self.assertEqual([True], l)
        self.assertTrue(self.fakeFile.closed)