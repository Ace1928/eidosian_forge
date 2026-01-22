from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
def onAuthenticated(self, xs):
    """
        Called when a component has successfully authenticated.

        Add the component to the routing table and establish a handler
        for a closed connection.
        """
    destination = xs.thisEntity.host
    self.router.addRoute(destination, xs)
    xs.addObserver(xmlstream.STREAM_END_EVENT, self.onConnectionLost, 0, destination, xs)