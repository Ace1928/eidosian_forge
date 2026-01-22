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
def test_savedEntryAfterAddHasKeyMismatch(self):
    """
        Even after a new entry has been added in memory but not yet saved, the
        L{HostKeyChanged} exception raised by L{KnownHostsFile.hasHostKey} has a
        C{lineno} attribute which indicates the 1-based line number of the
        offending entry in the underlying file when the given host key does not
        match the expected host key.
        """
    hostsFile = self.loadSampleHostsFile()
    hostsFile.addHostKey(b'www.example.com', Key.fromString(otherSampleKey))
    exception = self.assertRaises(HostKeyChanged, hostsFile.hasHostKey, b'www.twistedmatrix.com', Key.fromString(otherSampleKey))
    self.assertEqual(exception.lineno, 1)
    self.assertEqual(exception.path, hostsFile.savePath)