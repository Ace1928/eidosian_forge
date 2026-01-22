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
def verifyNonPresentKey(self):
    """
        Set up a test to verify a key that isn't present.  Return a 3-tuple of
        the UI, a list set up to collect the result of the verifyHostKey call,
        and the sample L{KnownHostsFile} being used.

        This utility method avoids returning a L{Deferred}, and records results
        in the returned list instead, because the events which get generated
        here are pre-recorded in the 'ui' object.  If the L{Deferred} in
        question does not fire, the it will fail quickly with an empty list.
        """
    hostsFile = self.loadSampleHostsFile()
    absentKey = Key.fromString(thirdSampleKey)
    ui = FakeUI()
    l = []
    d = hostsFile.verifyHostKey(ui, b'sample-host.example.com', b'4.3.2.1', absentKey)
    d.addBoth(l.append)
    self.assertEqual([], l)
    self.assertEqual(ui.promptText, b"The authenticity of host 'sample-host.example.com (4.3.2.1)' can't be established.\nRSA key fingerprint is SHA256:mS7mDBGhewdzJkaKRkx+wMjUdZb/GzvgcdoYjX5Js9I=.\nAre you sure you want to continue connecting (yes/no)? ")
    return (ui, l, hostsFile)