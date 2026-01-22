from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testCantForwardRequest(self):
    r = sip.Request('INVITE', 'sip:foo')
    r.addHeader('via', sip.Via('1.2.3.4').toString())
    r.addHeader('to', '<sip:joe@server.com>')
    self.proxy.locator = FailingLocator()
    self.proxy.datagramReceived(r.toString(), ('1.2.3.4', 5060))
    self.assertEqual(len(self.sent), 1)
    dest, m = self.sent[0]
    self.assertEqual((dest.host, dest.port), ('1.2.3.4', 5060))
    self.assertEqual(m.code, 404)
    self.assertEqual(m.headers['via'], ['SIP/2.0/UDP 1.2.3.4:5060'])