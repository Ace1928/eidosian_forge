from binascii import hexlify
from hashlib import sha1
from sys import intern
from typing import Optional, Tuple
from zope.interface import directlyProvides, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import ConnectionLost
from twisted.python import failure, log, randbytes
from twisted.words.protocols.jabber import error, ijabber, jid
from twisted.words.xish import domish, xmlstream
from twisted.words.xish.xmlstream import (
def upgradeWithIQResponseTracker(xs):
    """
    Enhances an XmlStream for iq response tracking.

    This makes an L{XmlStream} object provide L{IIQResponseTracker}. When a
    response is an error iq stanza, the deferred has its errback invoked with a
    failure that holds a L{StanzaError<error.StanzaError>} that is
    easier to examine.
    """

    def callback(iq):
        """
        Handle iq response by firing associated deferred.
        """
        if getattr(iq, 'handled', False):
            return
        try:
            d = xs.iqDeferreds[iq['id']]
        except KeyError:
            pass
        else:
            del xs.iqDeferreds[iq['id']]
            iq.handled = True
            if iq['type'] == 'error':
                d.errback(error.exceptionFromStanza(iq))
            else:
                d.callback(iq)

    def disconnected(_):
        """
        Make sure deferreds do not linger on after disconnect.

        This errbacks all deferreds of iq's for which no response has been
        received with a L{ConnectionLost} failure. Otherwise, the deferreds
        will never be fired.
        """
        iqDeferreds = xs.iqDeferreds
        xs.iqDeferreds = {}
        for d in iqDeferreds.values():
            d.errback(ConnectionLost())
    xs.iqDeferreds = {}
    xs.iqDefaultTimeout = getattr(xs, 'iqDefaultTimeout', None)
    xs.addObserver(xmlstream.STREAM_END_EVENT, disconnected)
    xs.addObserver('/iq[@type="result"]', callback)
    xs.addObserver('/iq[@type="error"]', callback)
    directlyProvides(xs, ijabber.IIQResponseTracker)