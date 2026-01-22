import itertools
from zope.interface import directlyProvides, implementer
from twisted.internet import error, interfaces
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import MemoryReactorClock
from twisted.python.failure import Failure
def bufferReceived(self, buf):
    if isinstance(buf, TLSNegotiation):
        assert self.tls is not None
        if self.tls.sent:
            self.tls.pretendToVerify(buf, self)
            self.tls = None
            b, self.tlsbuf = (self.tlsbuf, None)
            self.writeSequence(b)
            directlyProvides(self, interfaces.ISSLTransport)
        else:
            self.tls.readyToSend = True
    else:
        self.protocol.dataReceived(buf)