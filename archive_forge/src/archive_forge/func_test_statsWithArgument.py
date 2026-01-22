from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_statsWithArgument(self):
    """
        L{MemCacheProtocol.stats} takes an optional C{bytes} argument which,
        if specified, is sent along with the I{STAT} command.  The I{STAT}
        responses from the server are parsed as key/value pairs and returned
        as a C{dict} (as in the case where the argument is not specified).
        """
    return self._test(self.proto.stats(b'blah'), b'stats blah\r\n', b'STAT foo bar\r\nSTAT egg spam\r\nEND\r\n', {b'foo': b'bar', b'egg': b'spam'})