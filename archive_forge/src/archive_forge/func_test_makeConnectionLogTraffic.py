from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
def test_makeConnectionLogTraffic(self):
    """
        Setting logTraffic should set up raw data loggers.
        """
    self.factory.logTraffic = True
    self.xmlstream.dispatch(self.xmlstream, xmlstream.STREAM_CONNECTED_EVENT)
    self.assertNotIdentical(None, self.xmlstream.rawDataInFn)
    self.assertNotIdentical(None, self.xmlstream.rawDataOutFn)