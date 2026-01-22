from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
def test_routeDefault(self):
    """
        Test routing of a message using the default route.

        The default route is the one with L{None} as its key in the
        routing table. It is taken when there is no more specific route
        in the routing table that matches the stanza's destination.
        """
    component1 = XmlPipe()
    s2s = XmlPipe()
    router = component.Router()
    router.addRoute('component1.example.org', component1.sink)
    router.addRoute(None, s2s.sink)
    outgoing = []
    s2s.source.addObserver('/*', lambda element: outgoing.append(element))
    stanza = domish.Element((None, 'presence'))
    stanza['from'] = 'component1.example.org'
    stanza['to'] = 'example.com'
    component1.source.send(stanza)
    self.assertEqual([stanza], outgoing)