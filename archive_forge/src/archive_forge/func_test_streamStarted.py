from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
def test_streamStarted(self):
    """
        The received stream header should set several attributes.
        """
    observers = []

    def addOnetimeObserver(event, observerfn):
        observers.append((event, observerfn))
    xs = self.xmlstream
    xs.addOnetimeObserver = addOnetimeObserver
    xs.makeConnection(self)
    self.assertIdentical(None, xs.sid)
    self.assertFalse(xs._headerSent)
    xs.dataReceived("<stream:stream xmlns='jabber:component:accept' xmlns:stream='http://etherx.jabber.org/streams' to='component.example.org'>")
    self.assertEqual((0, 0), xs.version)
    self.assertNotIdentical(None, xs.sid)
    self.assertTrue(xs._headerSent)
    self.assertEqual(('/*', xs.authenticator.onElement), observers[-1])