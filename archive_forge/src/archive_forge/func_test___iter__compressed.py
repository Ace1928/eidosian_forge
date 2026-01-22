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
def test___iter__compressed(self):
    self.assertEqual({b'foo'}, set(self.get_populated_store(compressed=True).__iter__()))
    self.assertEqual({b'foo'}, set(self.get_populated_store(True, compressed=True).__iter__()))