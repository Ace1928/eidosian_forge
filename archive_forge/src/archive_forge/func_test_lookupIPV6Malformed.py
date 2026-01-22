from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase
def test_lookupIPV6Malformed(self) -> None:
    """
        Like L{test_lookupAddressMalformed}, but for
        L{hosts.Resolver.lookupIPV6Address}.
        """
    d = self.resolver.lookupIPV6Address(b'malformed')
    [answer], authority, additional = self.successResultOf(d)
    self.assertEqual(RRHeader(b'malformed', AAAA, IN, self.ttl, Record_AAAA('::5', self.ttl)), answer)