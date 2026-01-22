import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_MD5SessHashA2Auth(self):
    """
        L{calcHA2} accepts the C{'md5-sess'} algorithm and QOP of C{'auth'} and
        returns the same value as it does for the C{'md5'} algorithm.
        """
    self.test_MD5HashA2Auth(b'md5-sess')