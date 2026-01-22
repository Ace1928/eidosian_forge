import sys
from unittest import skipIf
from twisted.conch.error import ConchError
from twisted.conch.test import keydata
from twisted.internet.testing import StringTransport
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_getPublicKeyAgentFallback(self):
    """
        If an agent is present, but doesn't return a key,
        L{SSHUserAuthClient.getPublicKey} continue with the normal key lookup.
        """
    options = ConchOptions()
    options.identitys = [self.rsaFile.path]
    agent = SSHAgentClient()
    client = SSHUserAuthClient(b'user', options, None)
    client.keyAgent = agent
    key = client.getPublicKey()
    self.assertTrue(key.isPublic())
    self.assertEqual(key, self.rsaPublic)