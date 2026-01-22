from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase
def test_allAddresses(self) -> None:
    """
        L{searchFileForAll} returns a list of all addresses associated with the
        name passed to it.
        """
    hosts = self.path()
    hosts.setContent(b'127.0.0.1     foobar.example.com\n127.0.0.2     foobar.example.com\n::1           foobar.example.com\n')
    self.assertEqual(['127.0.0.1', '127.0.0.2', '::1'], searchFileForAll(hosts, b'foobar.example.com'))