import struct
from itertools import chain
from typing import Dict, List, Tuple
from twisted.conch.test.keydata import (
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessTerminated
from twisted.python import failure, log
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.python import components
def global_tcpip_forward(self, data):
    host, port = forwarding.unpackGlobal_tcpip_forward(data)
    try:
        listener = reactor.listenTCP(port, forwarding.SSHListenForwardingFactory(self.conn, (host, port), forwarding.SSHListenServerForwardingChannel), interface=host)
    except BaseException:
        log.err(None, 'something went wrong with remote->local forwarding')
        return 0
    else:
        self.listeners[host, port] = listener
        return 1