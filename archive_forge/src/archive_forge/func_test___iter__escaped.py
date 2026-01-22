import gzip
import os
from io import BytesIO
from ... import errors as errors
from ... import transactions, transport
from ...bzr.weave import WeaveFile
from ...errors import BzrError
from ...tests import TestCase, TestCaseInTempDir, TestCaseWithTransport
from ...transport.memory import MemoryTransport
from .store import TransportStore
from .store.text import TextStore
from .store.versioned import VersionedFileStore
def test___iter__escaped(self):
    self.vfstore = VersionedFileStore(MemoryTransport(), prefixed=True, escaped=True, versionedfile_class=WeaveFile)
    self.vfstore.get_scope = self.get_scope
    self._transaction = transactions.WriteTransaction()
    vf = self.vfstore.get_weave_or_empty(b' ', self._transaction)
    vf.add_lines(b'a', [], [])
    del vf
    self._transaction.finish()
    self.assertEqual([b' '], list(self.vfstore))