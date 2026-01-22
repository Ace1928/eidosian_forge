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
def test_savedEntryHasKeyMismatch(self):
    """
        L{KnownHostsFile.hasHostKey} raises L{HostKeyChanged} if the host key is
        present in the underlying file, but different from the expected one.
        The resulting exception should have an C{offendingEntry} indicating the
        given entry.
        """
    hostsFile = self.loadSampleHostsFile()
    entries = list(hostsFile.iterentries())
    exception = self.assertRaises(HostKeyChanged, hostsFile.hasHostKey, b'www.twistedmatrix.com', Key.fromString(otherSampleKey))
    self.assertEqual(exception.offendingEntry, entries[0])
    self.assertEqual(exception.lineno, 1)
    self.assertEqual(exception.path, hostsFile.savePath)