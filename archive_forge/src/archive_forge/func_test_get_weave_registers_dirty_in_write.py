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
def test_get_weave_registers_dirty_in_write(self):
    self._transaction = transactions.WriteTransaction()
    vf = self.vfstore.get_weave_or_empty(b'id', self._transaction)
    self._transaction.finish()
    self._transaction = None
    self.assertRaises(errors.OutSideTransaction, vf.add_lines, b'b', [], [])
    self._transaction = transactions.WriteTransaction()
    vf = self.vfstore.get_weave(b'id', self._transaction)
    self._transaction.finish()
    self._transaction = None
    self.assertRaises(errors.OutSideTransaction, vf.add_lines, b'b', [], [])