from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
def onHandshake(self, handshake):
    """
        Called upon receiving the handshake request.

        This checks that the given hash in C{handshake} is equal to a
        calculated hash, responding with a handshake reply or a stream error.
        If the handshake was ok, the stream is authorized, and  XML Stanzas may
        be exchanged.
        """
    calculatedHash = xmlstream.hashPassword(self.xmlstream.sid, str(self.secret))
    if handshake != calculatedHash:
        exc = error.StreamError('not-authorized', text='Invalid hash')
        self.xmlstream.sendStreamError(exc)
    else:
        self.xmlstream.send('<handshake/>')
        self.xmlstream.dispatch(self.xmlstream, xmlstream.STREAM_AUTHD_EVENT)