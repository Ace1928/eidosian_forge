import os.path
from errno import ENOSYS
from struct import pack
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
import hamcrest
from twisted.conch.error import ConchError, HostKeyChanged, UserRejectedKey
from twisted.conch.interfaces import IConchUser
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.portal import Portal
from twisted.internet.address import IPv4Address
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import (
from twisted.internet.interfaces import IAddress, IStreamClientEndpoint
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import (
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
from twisted.test.iosim import FakeTransport, connect
def loseConnectionToServer(self, server, client, protocol, pump):
    """
        Lose the connection to a server and pump the L{IOPump} sufficiently for
        the client to handle the lost connection. Asserts that the client
        disconnects its transport.

        @param server: The SSH server protocol over which C{protocol} is
            running.
        @type server: L{IProtocol} provider

        @param client: The SSH client protocol over which C{protocol} is
            running.
        @type client: L{IProtocol} provider

        @param protocol: The protocol created by calling connect on the ssh
            endpoint under test.
        @type protocol: L{IProtocol} provider

        @param pump: The L{IOPump} connecting client to server.
        @type pump: L{IOPump}
        """
    closed = self.record(server, protocol, 'closed', noArgs=True)
    protocol.transport.loseConnection()
    pump.pump()
    self.assertEqual([None], closed)
    pump.pump()
    client.transport.reportDisconnect()