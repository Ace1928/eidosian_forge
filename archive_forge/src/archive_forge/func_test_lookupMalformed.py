from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase
def test_lookupMalformed(self) -> None:
    """
        L{hosts.Resolver.lookupAddress} returns a L{Deferred} which fires with
        the valid addresses from the hosts file, ignoring any entries that
        aren't valid IP addresses.
        """
    d = self.resolver.lookupAddress(b'malformed')
    [answer], authority, additional = self.successResultOf(d)
    self.assertEqual(RRHeader(b'malformed', A, IN, self.ttl, Record_A('1.1.1.5', self.ttl)), answer)