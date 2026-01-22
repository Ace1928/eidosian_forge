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
def test_inKnownHostsKeyChanged(self):
    """
        L{default.isInKnownHosts} should return C{2} when a host with a key
        other than the given one is in the known hosts file.
        """
    host = self.hashedEntries[b'4.3.2.1'].toString().split()[0]
    r = default.isInKnownHosts(host, Key.fromString(otherSampleKey).blob(), {'known-hosts': FilePath(self.hostsOption).path})
    self.assertEqual(2, r)