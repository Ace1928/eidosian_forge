from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
def test_connectionInitialized(self):
    """
        Make sure a new stream is added to the routing table.
        """
    self.xmlstream.dispatch(self.xmlstream, xmlstream.STREAM_AUTHD_EVENT)
    self.assertIn('component.example.org', self.router.routes)
    self.assertIdentical(self.xmlstream, self.router.routes['component.example.org'])