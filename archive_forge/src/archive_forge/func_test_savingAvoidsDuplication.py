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
def test_savingAvoidsDuplication(self):
    """
        L{KnownHostsFile.save} only writes new entries to the save path, not
        entries which were added and already written by a previous call to
        C{save}.
        """
    path = FilePath(self.mktemp())
    knownHosts = KnownHostsFile(path)
    entry = knownHosts.addHostKey(b'some.example.com', Key.fromString(sampleKey))
    knownHosts.save()
    knownHosts.save()
    knownHosts = KnownHostsFile.fromPath(path)
    self.assertEqual([entry], list(knownHosts.iterentries()))