import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_responseWithoutClientIP(self):
    """
        L{DigestCredentialFactory.decode} accepts a digest challenge response
        even if the client address it is passed is L{None}.
        """
    challenge = self.credentialFactory.getChallenge(None)
    nc = b'00000001'
    clientResponse = self.formatResponse(nonce=challenge['nonce'], response=self.getDigestResponse(challenge, nc), nc=nc, opaque=challenge['opaque'])
    creds = self.credentialFactory.decode(clientResponse, self.method, None)
    self.assertTrue(creds.checkPassword(self.password))
    self.assertFalse(creds.checkPassword(self.password + b'wrong'))