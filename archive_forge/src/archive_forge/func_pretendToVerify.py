import itertools
from zope.interface import directlyProvides, implementer
from twisted.internet import error, interfaces
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import MemoryReactorClock
from twisted.python.failure import Failure
def pretendToVerify(self, other, tpt):
    if not self.obj.iosimVerify(other.obj):
        tpt.disconnectReason = NativeOpenSSLError()
        tpt.loseConnection()