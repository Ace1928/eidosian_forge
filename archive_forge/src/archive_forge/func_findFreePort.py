import socket
from gc import collect
from typing import Optional
from weakref import ref
from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.interfaces import IConnector, IReactorFDSet
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.reactormixins import needsRunningReactor
from twisted.python import context, log
from twisted.python.failure import Failure
from twisted.python.log import ILogContext, err, msg
from twisted.python.runtime import platform
from twisted.test.test_tcp import ClosingProtocol
from twisted.trial.unittest import SkipTest
def findFreePort(interface='127.0.0.1', family=socket.AF_INET, type=socket.SOCK_STREAM):
    """
    Ask the platform to allocate a free port on the specified interface, then
    release the socket and return the address which was allocated.

    @param interface: The local address to try to bind the port on.
    @type interface: C{str}

    @param type: The socket type which will use the resulting port.

    @return: A two-tuple of address and port, like that returned by
        L{socket.getsockname}.
    """
    addr = socket.getaddrinfo(interface, 0)[0][4]
    probe = socket.socket(family, type)
    try:
        probe.bind(addr)
        if family == socket.AF_INET6:
            sockname = probe.getsockname()
            hostname = socket.getnameinfo(sockname, socket.NI_NUMERICHOST | socket.NI_NUMERICSERV)[0]
            return (hostname, sockname[1])
        else:
            return probe.getsockname()
    finally:
        probe.close()