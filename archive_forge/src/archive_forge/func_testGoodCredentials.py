from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
def testGoodCredentials(self):
    goodCreds = [credentials.UsernamePassword(u, p) for u, p in self.users]
    d = defer.gatherResults([self.db.requestAvatarId(c) for c in goodCreds])
    d.addCallback(self.assertEqual, [u for u, p in self.users])
    return d