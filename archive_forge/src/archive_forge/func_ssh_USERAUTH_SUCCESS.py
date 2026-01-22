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
def ssh_USERAUTH_SUCCESS(self, packet):
    """
        We received a MSG_USERAUTH_SUCCESS.  The server has accepted our
        authentication, so start the next service.
        """
    self.transport.setService(self.instance)