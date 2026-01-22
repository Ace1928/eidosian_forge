import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_responseWithCommaURI(self):
    """
        L{DigestCredentialFactory.decode} accepts a digest challenge response
        which quotes the values of its fields and includes a C{b","} in the URI
        field.
        """
    self.uri = b'/some,path/'
    self.test_response(True)