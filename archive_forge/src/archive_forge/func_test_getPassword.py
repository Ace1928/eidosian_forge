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
@skipIf(doSkip, skipReason)
def test_getPassword(self):
    """
        Get the password using
        L{twisted.conch.client.default.SSHUserAuthClient.getPassword}
        """

    class FakeTransport:

        def __init__(self, host):
            self.transport = self
            self.host = host

        def getPeer(self):
            return self
    options = ConchOptions()
    client = SSHUserAuthClient(b'user', options, None)
    client.transport = FakeTransport('127.0.0.1')

    def getpass(prompt):
        self.assertEqual(prompt, "user@127.0.0.1's password: ")
        return 'bad password'
    self.patch(default.getpass, 'getpass', getpass)
    d = client.getPassword()
    d.addCallback(self.assertEqual, b'bad password')
    return d