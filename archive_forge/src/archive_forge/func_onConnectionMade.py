from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
def onConnectionMade(self, xs):
    """
        Called when a component connection was made.

        This enables traffic debugging on incoming streams.
        """
    xs.serial = self.serial
    self.serial += 1

    def logDataIn(buf):
        log.msg('RECV (%d): %r' % (xs.serial, buf))

    def logDataOut(buf):
        log.msg('SEND (%d): %r' % (xs.serial, buf))
    if self.logTraffic:
        xs.rawDataInFn = logDataIn
        xs.rawDataOutFn = logDataOut
    xs.addObserver(xmlstream.STREAM_ERROR_EVENT, self.onError)