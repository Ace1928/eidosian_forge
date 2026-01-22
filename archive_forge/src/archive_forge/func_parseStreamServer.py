from typing import Mapping, Tuple
from zope.interface import implementer
from twisted.internet import interfaces
from twisted.internet.endpoints import (
from twisted.plugin import IPlugin
from . import proxyEndpoint
def parseStreamServer(self, reactor: interfaces.IReactorCore, *args: object, **kwargs: object) -> _WrapperServerEndpoint:
    """
        Parse a stream server endpoint from a reactor and string-only arguments
        and keyword arguments.

        @param reactor: The reactor.

        @param args: The parsed string arguments.

        @param kwargs: The parsed keyword arguments.

        @return: a stream server endpoint
        @rtype: L{IStreamServerEndpoint}
        """
    subdescription = unparseEndpoint(args, kwargs)
    wrappedEndpoint = serverFromString(reactor, subdescription)
    return proxyEndpoint(wrappedEndpoint)