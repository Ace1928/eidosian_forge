from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_getInChunks(self):
    """
        If the value retrieved by a C{get} arrive in chunks, the protocol
        is able to reconstruct it and to produce the good value.
        """
    d = self.proto.get(b'foo')
    d.addCallback(self.assertEqual, (0, b'0123456789'))
    self.assertEqual(self.transport.value(), b'get foo\r\n')
    self.proto.dataReceived(b'VALUE foo 0 10\r\n0123456')
    self.proto.dataReceived(b'789')
    self.proto.dataReceived(b'\r\nEND')
    self.proto.dataReceived(b'\r\n')
    return d