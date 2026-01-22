from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_invalidGetResponse(self):
    """
        If the value returned doesn't match the expected key of the current
        C{get} command, an error is raised in L{MemCacheProtocol.dataReceived}.
        """
    self.proto.get(b'foo')
    self.assertRaises(RuntimeError, self.proto.dataReceived, b'VALUE bar 0 7\r\nspamegg\r\nEND\r\n')