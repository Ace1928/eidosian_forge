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
def test_has_suffixed_no_base(self):
    my_store = self.get_populated_store()
    self.assertEqual(False, my_store.has_id(b'missing'))
    my_store = self.get_populated_store(True)
    self.assertEqual(False, my_store.has_id(b'missing'))