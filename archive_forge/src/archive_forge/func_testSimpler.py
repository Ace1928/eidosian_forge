from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testSimpler(self):
    v = sip.Via('example.com')
    self.checkRoundtrip(v)