from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers

        Construct and return an L{IStreamClientEndpoint} for the outgoing
        request's connection.

        @param uri: The URI of the request.
        @type uri: L{twisted.web.client.URI}

        @return: An endpoint which will have its C{connect} method called to
            issue the request.
        @rtype: an L{IStreamClientEndpoint} provider

        @raises twisted.internet.error.SchemeNotSupported: If the given
            URI's scheme cannot be handled by this factory.
        