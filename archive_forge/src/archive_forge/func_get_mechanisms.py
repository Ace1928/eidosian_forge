import re
from base64 import b64decode, b64encode
from twisted.internet import defer
from twisted.words.protocols.jabber import sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def get_mechanisms(xs):
    """
    Parse the SASL feature to extract the available mechanism names.
    """
    mechanisms = []
    for element in xs.features[NS_XMPP_SASL, 'mechanisms'].elements():
        if element.name == 'mechanism':
            mechanisms.append(str(element))
    return mechanisms