import hashlib
from binascii import hexlify
from hmac import HMAC
from twisted.cred.credentials import CramMD5Credentials, IUsernameHashedPassword
from twisted.trial.unittest import TestCase
def test_checkPassword(self) -> None:
    """
        When a valid response (which is a hex digest of the challenge that has
        been encrypted by the user's shared secret) is set on the
        L{CramMD5Credentials} that created the challenge, and C{checkPassword}
        is called with the user's shared secret, it will return L{True}.
        """
    c = CramMD5Credentials()
    chal = c.getChallenge()
    c.response = hexlify(HMAC(b'secret', chal, digestmod=hashlib.md5).digest())
    self.assertTrue(c.checkPassword(b'secret'))