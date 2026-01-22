from twisted.cred.credentials import (
from twisted.cred.test.test_cred import _uhpVersion
from twisted.trial.unittest import TestCase
def test_correctPassword(self) -> None:
    """
        Calling C{checkPassword} on a L{UsernameHashedPassword} will return
        L{True} when the password given is the password on the object.
        """
    UsernameHashedPassword = self.getDeprecatedModuleAttribute('twisted.cred.credentials', 'UsernameHashedPassword', _uhpVersion)
    creds = UsernameHashedPassword(b'user', b'pass')
    self.assertTrue(creds.checkPassword(b'pass'))