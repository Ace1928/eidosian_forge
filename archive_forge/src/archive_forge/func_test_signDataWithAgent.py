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
def test_signDataWithAgent(self):
    """
        When connected to an agent, L{SSHUserAuthClient} can use it to
        request signatures of particular data with a particular L{Key}.
        """
    client = SSHUserAuthClient(b'user', ConchOptions(), None)
    agent = SSHAgentClient()
    transport = StringTransport()
    agent.makeConnection(transport)
    client.keyAgent = agent
    cleartext = b'Sign here'
    client.signData(self.rsaPublic, cleartext)
    self.assertEqual(transport.value(), b'\x00\x00\x01-\r\x00\x00\x01\x17' + self.rsaPublic.blob() + b'\x00\x00\x00\t' + cleartext + b'\x00\x00\x00\x00')