from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_emptyGet(self):
    """
        Test getting a non-available key: it succeeds but return L{None} as
        value and C{0} as flag.
        """
    return self._test(self.proto.get(b'foo'), b'get foo\r\n', b'END\r\n', (0, None))