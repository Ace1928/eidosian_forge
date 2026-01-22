from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
def associateWithStream(self, xs):
    """
        Associate the authenticator with a stream.

        This sets the stream's version to 0.0, because the XEP-0114 component
        protocol was not designed for XMPP 1.0.
        """
    xs.version = (0, 0)
    xmlstream.ListenAuthenticator.associateWithStream(self, xs)