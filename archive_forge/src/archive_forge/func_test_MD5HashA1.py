import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_MD5HashA1(self, _algorithm=b'md5', _hash=md5):
    """
        L{calcHA1} accepts the C{'md5'} algorithm and returns an MD5 hash of
        its parameters, excluding the nonce and cnonce.
        """
    nonce = b'abc123xyz'
    hashA1 = calcHA1(_algorithm, self.username, self.realm, self.password, nonce, self.cnonce)
    a1 = b':'.join((self.username, self.realm, self.password))
    expected = hexlify(_hash(a1).digest())
    self.assertEqual(hashA1, expected)