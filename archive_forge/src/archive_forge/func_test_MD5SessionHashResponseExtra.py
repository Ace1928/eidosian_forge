import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_MD5SessionHashResponseExtra(self):
    """
        L{calcResponse} accepts the C{'md5-sess'} algorithm and returns an MD5
        hash of its parameters, including the nonce count, client nonce, and
        QoP value if they are specified.
        """
    self.test_MD5HashResponseExtra(b'md5-sess')