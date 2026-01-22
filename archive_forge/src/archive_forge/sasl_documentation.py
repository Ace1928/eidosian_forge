import re
from base64 import b64decode, b64encode
from twisted.internet import defer
from twisted.words.protocols.jabber import sasl_mechanisms, xmlstream
from twisted.words.xish import domish

        Clean up observers, parse the failure and errback the deferred.

        @param failure: the failure protocol element. Holds details on
                        the error condition.
        @type failure: L{domish.Element}
        