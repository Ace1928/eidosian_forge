import re
from base64 import b64decode, b64encode
from twisted.internet import defer
from twisted.words.protocols.jabber import sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def fromBase64(s):
    """
    Decode base64 encoded string.

    This helper performs regular decoding of a base64 encoded string, but also
    rejects any characters that are not in the base64 alphabet and padding
    occurring elsewhere from the last or last two characters, as specified in
    section 14.9 of RFC 3920. This safeguards against various attack vectors
    among which the creation of a covert channel that "leaks" information.
    """
    if base64Pattern.match(s) is None:
        raise SASLIncorrectEncodingError()
    try:
        return b64decode(s)
    except Exception as e:
        raise SASLIncorrectEncodingError(str(e))