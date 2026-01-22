from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_getMultipleWithEmpty(self):
    """
        When L{MemCacheProtocol.getMultiple} is called with non-available keys,
        the corresponding tuples are (0, None).
        """
    return self._test(self.proto.getMultiple([b'foo', b'cow']), b'get foo cow\r\n', b'VALUE cow 1 3\r\nbar\r\nEND\r\n', {b'cow': (1, b'bar'), b'foo': (0, None)})