import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_responseWithoutQuotes(self):
    """
        L{DigestCredentialFactory.decode} accepts a digest challenge response
        which does not quote the values of its fields and parses it into an
        L{IUsernameHashedPassword} provider in the same way it would a
        response which included quoted field values.
        """
    self.test_response(False)