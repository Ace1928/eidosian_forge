import itertools
from zope.interface import directlyProvides, implementer
from twisted.internet import error, interfaces
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import MemoryReactorClock
from twisted.python.failure import Failure
def makeFakeServer(serverProtocol):
    """
    Create and return a new in-memory transport hooked up to the given protocol.

    @param serverProtocol: The server protocol to use.
    @type serverProtocol: L{IProtocol} provider

    @return: The transport.
    @rtype: L{FakeTransport}
    """
    return FakeTransport(serverProtocol, isServer=True)