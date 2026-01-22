from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers
def setPreviousResponse(response):
    """
        Set the reference to the previous L{IResponse}.

        The value of the previous response can be read via
        L{IResponse.previousResponse}.
        """