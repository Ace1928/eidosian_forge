from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
def test_onElementNotHandshake(self):
    """
        Reject elements that are not handshakes
        """
    handshakes = []
    streamErrors = []
    xs = self.xmlstream
    xs.authenticator.onHandshake = handshakes.append
    xs.sendStreamError = streamErrors.append
    element = domish.Element(('jabber:component:accept', 'message'))
    xs.authenticator.onElement(element)
    self.assertFalse(handshakes)
    self.assertEqual('not-authorized', streamErrors[-1].condition)