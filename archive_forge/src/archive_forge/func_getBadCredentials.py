from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
def getBadCredentials(self):
    for u, p in [(b'user1', b'password3'), (b'user2', b'password1'), (b'bloof', b'blarf')]:
        yield self.credClass(u, self.networkHash(p))