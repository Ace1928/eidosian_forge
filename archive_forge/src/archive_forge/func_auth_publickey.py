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
def auth_publickey(self):
    """
        Try to authenticate with a public key.  Ask the user for a public key;
        if the user has one, send the request to the server and return True.
        Otherwise, return False.

        @rtype: L{bool}
        """
    d = defer.maybeDeferred(self.getPublicKey)
    d.addBoth(self._cbGetPublicKey)
    return d