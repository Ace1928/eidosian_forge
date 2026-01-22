from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testSimpleResponse(self):
    l = self.l
    self.feedMessage(response1)
    self.assertEqual(len(l), 1)
    m = l[0]
    self.assertEqual(m.code, 200)
    self.assertEqual(m.phrase, 'OK')
    self.assertEqual(m.headers, {'from': ['foo'], 'to': ['bar'], 'content-length': ['0']})
    self.assertEqual(m.body, '')
    self.assertEqual(m.finished, 1)