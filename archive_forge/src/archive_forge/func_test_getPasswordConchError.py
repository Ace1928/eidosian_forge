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
def test_getPasswordConchError(self):
    """
        Get the password using
        L{twisted.conch.client.default.SSHUserAuthClient.getPassword}
        and trigger a {twisted.conch.error import ConchError}.
        """
    options = ConchOptions()
    client = SSHUserAuthClient(b'user', options, None)

    def getpass(prompt):
        raise KeyboardInterrupt('User pressed CTRL-C')
    self.patch(default.getpass, 'getpass', getpass)
    stdout, stdin = (sys.stdout, sys.stdin)
    d = client.getPassword(b'?')

    @d.addErrback
    def check_sys(fail):
        self.assertEqual([stdout, stdin], [sys.stdout, sys.stdin])
        return fail
    self.assertFailure(d, ConchError)