import re
from base64 import b64decode, b64encode
from twisted.internet import defer
from twisted.words.protocols.jabber import sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def sendResponse(self, data=b''):
    """
        Send response to a challenge.

        @param data: client response.
        @type data: L{bytes}.
        """
    response = domish.Element((NS_XMPP_SASL, 'response'))
    if data:
        response.addContent(b64encode(data).decode('ascii'))
    self.xmlstream.send(response)