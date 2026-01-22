import signal
from os.path import expanduser
from struct import unpack
from zope.interface import Interface, implementer
from twisted.conch.client.agent import SSHAgentClient
from twisted.conch.client.default import _KNOWN_HOSTS
from twisted.conch.client.knownhosts import ConsoleUI, KnownHostsFile
from twisted.conch.ssh.channel import SSHChannel
from twisted.conch.ssh.common import NS, getNS
from twisted.conch.ssh.connection import SSHConnection
from twisted.conch.ssh.keys import Key
from twisted.conch.ssh.transport import SSHClientTransport
from twisted.conch.ssh.userauth import SSHUserAuthClient
from twisted.internet.defer import CancelledError, Deferred, succeed
from twisted.internet.endpoints import TCP4ClientEndpoint, connectProtocol
from twisted.internet.error import ConnectionDone, ProcessTerminated
from twisted.internet.interfaces import IStreamClientEndpoint
from twisted.internet.protocol import Factory
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
@classmethod
def existingConnection(cls, connection, command):
    """
        Create and return a new endpoint which will try to open a new channel
        on an existing SSH connection and run a command over it.  It will
        B{not} close the connection if there is a problem executing the command
        or after the command finishes.

        @param connection: An existing connection to an SSH server.
        @type connection: L{SSHConnection}

        @param command: See L{SSHCommandClientEndpoint.newConnection}'s
            C{command} parameter.
        @type command: L{bytes}

        @return: A new instance of C{cls} (probably
            L{SSHCommandClientEndpoint}).
        """
    helper = _ExistingConnectionHelper(connection)
    return cls(helper, command)