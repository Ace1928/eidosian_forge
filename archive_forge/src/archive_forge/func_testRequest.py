from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testRequest(self):
    r = sip.Request('INVITE', 'sip:foo')
    r.addHeader('foo', 'bar')
    self.assertEqual(r.toString(), 'INVITE sip:foo SIP/2.0\r\nFoo: bar\r\n\r\n')