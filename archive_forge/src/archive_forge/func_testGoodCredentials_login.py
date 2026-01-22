from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
def testGoodCredentials_login(self):
    goodCreds = [credentials.UsernamePassword(u, p) for u, p in self.users]
    d = defer.gatherResults([self.port.login(c, None, ITestable) for c in goodCreds])
    d.addCallback(lambda x: [a.original.name for i, a, l in x])
    d.addCallback(self.assertEqual, [u for u, p in self.users])
    return d