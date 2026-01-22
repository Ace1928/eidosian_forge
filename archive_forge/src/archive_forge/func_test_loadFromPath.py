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
def test_loadFromPath(self):
    """
        Loading a L{KnownHostsFile} from a path with six entries in it will
        result in a L{KnownHostsFile} object with six L{IKnownHostEntry}
        providers in it.
        """
    hostsFile = self.loadSampleHostsFile()
    self.assertEqual(6, len(list(hostsFile.iterentries())))