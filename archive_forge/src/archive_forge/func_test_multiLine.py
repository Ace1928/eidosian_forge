from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def test_multiLine(self):
    """
        A header may be split across multiple lines.  Subsequent lines begin
        with C{" "} or C{"\\t"}.
        """
    l = self.l
    self.feedMessage(response_multiline)
    self.assertEqual(len(l), 1)
    m = l[0]
    self.assertEqual(m.headers['via'][0], 'SIP/2.0/UDP server10.biloxi.com;branch=z9hG4bKnashds8;received=192.0.2.3')
    self.assertEqual(m.headers['via'][1], 'SIP/2.0/UDP bigbox3.site3.atlanta.com;branch=z9hG4bK77ef4c2312983.1;received=192.0.2.2')
    self.assertEqual(m.headers['via'][2], 'SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds ;received=192.0.2.1')