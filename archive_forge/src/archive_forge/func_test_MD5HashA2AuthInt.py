import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_MD5HashA2AuthInt(self, _algorithm=b'md5', _hash=md5):
    """
        L{calcHA2} accepts the C{'md5'} algorithm and returns an MD5 hash of
        its arguments, including the entity hash for QOP of C{'auth-int'}.
        """
    method = b'GET'
    hentity = b'foobarbaz'
    hashA2 = calcHA2(_algorithm, method, self.uri, b'auth-int', hentity)
    a2 = method + b':' + self.uri + b':' + hentity
    expected = hexlify(_hash(a2).digest())
    self.assertEqual(hashA2, expected)