from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testResponseWrongVia(self):
    r = sip.Response(200)
    r.addHeader('via', sip.Via('foo.com').toString())
    self.proxy.datagramReceived(r.toString(), ('1.1.1.1', 5060))
    self.assertEqual(len(self.sent), 0)