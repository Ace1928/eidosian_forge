from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers
def getCookie(key):
    """
        Get a cookie that was sent from the network.

        @type key: L{bytes}
        @param key: The name of the cookie to get.

        @rtype: L{bytes} or L{None}
        @returns: The value of the specified cookie, or L{None} if that cookie
            was not present in the request.
        """