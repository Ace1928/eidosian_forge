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
def test_agentGetPublicKey(self):
    """
        L{SSHUserAuthClient} looks up public keys from the agent using the
        L{SSHAgentClient} class.  That L{SSHAgentClient.getPublicKey} returns a
        L{Key} object with one of the public keys in the agent.  If no more
        keys are present, it returns L{None}.
        """
    agent = SSHAgentClient()
    agent.blobs = [self.rsaPublic.blob()]
    key = agent.getPublicKey()
    self.assertTrue(key.isPublic())
    self.assertEqual(key, self.rsaPublic)
    self.assertIsNone(agent.getPublicKey())