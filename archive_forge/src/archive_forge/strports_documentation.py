from typing import Optional, cast
from twisted.application.internet import StreamServerEndpointService
from twisted.internet import endpoints, interfaces

    Listen on a port corresponding to a description.

    @param description: The description of the connecting port, in the syntax
        described by L{twisted.internet.endpoints.serverFromString}.
    @type description: L{str}

    @param factory: The protocol factory which will build protocols on
        connection.
    @type factory: L{twisted.internet.interfaces.IProtocolFactory}

    @rtype: L{twisted.internet.interfaces.IListeningPort}
    @return: the port corresponding to a description of a reliable virtual
        circuit server.

    @see: L{twisted.internet.endpoints.serverFromString}
    