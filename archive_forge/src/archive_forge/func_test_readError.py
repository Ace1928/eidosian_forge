from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase
def test_readError(self) -> None:
    """
        If there is an error reading the contents of the hosts file,
        L{searchFileForAll} returns an empty list.
        """
    self.assertEqual([], searchFileForAll(self.path(), b'example.com'))