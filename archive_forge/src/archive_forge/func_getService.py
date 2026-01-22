import random
from itertools import chain
from typing import Dict, List, Optional, Tuple
from twisted.conch import error
from twisted.conch.ssh import _kex, connection, transport, userauth
from twisted.internet import protocol
from twisted.logger import Logger
def getService(self, transport, service):
    """
        Return a class to use as a service for the given transport.

        @type transport:    L{transport.SSHServerTransport}
        @type service:      L{bytes}
        @rtype:             subclass of L{service.SSHService}
        """
    if service == b'ssh-userauth' or hasattr(transport, 'avatar'):
        return self.services[service]