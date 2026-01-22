import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_MD5SessionHashA1(self):
    """
        L{calcHA1} accepts the C{'md5-sess'} algorithm and returns an MD5 hash
        of its parameters, including the nonce and cnonce.
        """
    nonce = b'xyz321abc'
    hashA1 = calcHA1(b'md5-sess', self.username, self.realm, self.password, nonce, self.cnonce)
    a1 = self.username + b':' + self.realm + b':' + self.password
    ha1 = hexlify(md5(a1).digest())
    a1 = ha1 + b':' + nonce + b':' + self.cnonce
    expected = hexlify(md5(a1).digest())
    self.assertEqual(hashA1, expected)