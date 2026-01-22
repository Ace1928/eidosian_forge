from typing import Optional, Union
from twisted.internet import interfaces
from twisted.internet.endpoints import _WrapperServerEndpoint
from twisted.protocols import policies
from . import _info
from ._exceptions import InvalidProxyHeader
from ._v1parser import V1Parser
from ._v2parser import V2Parser
def proxyEndpoint(wrappedEndpoint: interfaces.IStreamServerEndpoint) -> _WrapperServerEndpoint:
    """
    Wrap an endpoint with PROXY protocol support, so that the transport's
    C{getHost} and C{getPeer} methods reflect the attributes of the proxied
    connection rather than the underlying connection.

    @param wrappedEndpoint: The underlying listening endpoint.
    @type wrappedEndpoint: L{IStreamServerEndpoint}

    @return: a new listening endpoint that speaks the PROXY protocol.
    @rtype: L{IStreamServerEndpoint}
    """
    return _WrapperServerEndpoint(wrappedEndpoint, HAProxyWrappingFactory)