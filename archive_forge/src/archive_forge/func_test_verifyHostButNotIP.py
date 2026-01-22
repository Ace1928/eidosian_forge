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
def test_verifyHostButNotIP(self):
    """
        L{default.verifyHostKey} should return a L{Deferred} which fires with
        C{1} when passed a host which matches with an IP is not present in its
        known_hosts file, and should also warn the user that it has added the
        IP address.
        """
    l = []
    default.verifyHostKey(self.fakeTransport, b'8.7.6.5', sampleKey, b'Fingerprint not required.').addCallback(l.append)
    self.assertEqual(["Warning: Permanently added the RSA host key for IP address '8.7.6.5' to the list of known hosts."], self.fakeFile.outchunks)
    self.assertEqual([1], l)
    knownHostsFile = KnownHostsFile.fromPath(FilePath(self.hostsOption))
    self.assertTrue(knownHostsFile.hasHostKey(b'8.7.6.5', Key.fromString(sampleKey)))