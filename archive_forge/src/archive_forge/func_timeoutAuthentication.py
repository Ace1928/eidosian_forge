import struct
from twisted.conch import error, interfaces
from twisted.conch.ssh import keys, service, transport
from twisted.conch.ssh.common import NS, getNS
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, reactor
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString
def timeoutAuthentication(self):
    """
        Called when the user has timed out on authentication.  Disconnect
        with a DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE message.
        """
    self._cancelLoginTimeout = None
    self.transport.sendDisconnect(transport.DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE, b'you took too long')