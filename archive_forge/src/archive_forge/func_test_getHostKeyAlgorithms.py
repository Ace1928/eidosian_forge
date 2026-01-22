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
def test_getHostKeyAlgorithms(self):
    """
        For a given host, get the host key algorithms for that
        host in the known_hosts file.
        """
    hostsFile = self.loadSampleHostsFile()
    hostsFile.addHostKey(b'www.twistedmatrix.com', Key.fromString(otherSampleKey))
    hostsFile.addHostKey(b'www.twistedmatrix.com', Key.fromString(ecdsaSampleKey))
    hostsFile.save()
    options = {}
    options['known-hosts'] = hostsFile.savePath.path
    algorithms = default.getHostKeyAlgorithms(b'www.twistedmatrix.com', options)
    expectedAlgorithms = [b'ssh-rsa', b'ecdsa-sha2-nistp256']
    self.assertEqual(algorithms, expectedAlgorithms)