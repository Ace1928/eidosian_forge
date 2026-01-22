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
def test_verifyNonPresentECKey(self):
    """
        Set up a test to verify an ECDSA key that isn't present.
        Return a 3-tuple of the UI, a list set up to collect the result
        of the verifyHostKey call, and the sample L{KnownHostsFile} being used.
        """
    ecObj = Key._fromECComponents(x=keydata.ECDatanistp256['x'], y=keydata.ECDatanistp256['y'], privateValue=keydata.ECDatanistp256['privateValue'], curve=keydata.ECDatanistp256['curve'])
    hostsFile = self.loadSampleHostsFile()
    ui = FakeUI()
    l = []
    d = hostsFile.verifyHostKey(ui, b'sample-host.example.com', b'4.3.2.1', ecObj)
    d.addBoth(l.append)
    self.assertEqual([], l)
    self.assertEqual(ui.promptText, b"The authenticity of host 'sample-host.example.com (4.3.2.1)' can't be established.\nECDSA key fingerprint is SHA256:fJnSpgCcYoYYsaBbnWj1YBghGh/QTDgfe4w4U5M5tEo=.\nAre you sure you want to continue connecting (yes/no)? ")