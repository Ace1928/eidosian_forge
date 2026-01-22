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
def test_saveResetsClobberState(self):
    """
        After L{KnownHostsFile.save} is used once with an instance initialized
        by the default initializer, contents of the save path are respected and
        preserved.
        """
    hostsFile = KnownHostsFile(self.pathWithContent(sampleHashedLine))
    preSave = hostsFile.addHostKey(b'www.example.com', Key.fromString(otherSampleKey))
    hostsFile.save()
    postSave = hostsFile.addHostKey(b'another.example.com', Key.fromString(thirdSampleKey))
    hostsFile.save()
    self.assertEqual([preSave, postSave], list(hostsFile.iterentries()))