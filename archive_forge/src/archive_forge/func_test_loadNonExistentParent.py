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
def test_loadNonExistentParent(self):
    """
        Loading a L{KnownHostsFile} from a path whose parent directory does not
        exist should result in an empty L{KnownHostsFile} that will save back
        to that path, creating its parent directory(ies) in the process.
        """
    thePath = FilePath(self.mktemp())
    knownHostsPath = thePath.child('foo').child(b'known_hosts')
    knownHostsFile = KnownHostsFile.fromPath(knownHostsPath)
    knownHostsFile.save()
    knownHostsPath.restat(False)
    self.assertTrue(knownHostsPath.exists())