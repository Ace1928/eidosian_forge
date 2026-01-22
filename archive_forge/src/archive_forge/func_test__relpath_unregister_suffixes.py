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
def test__relpath_unregister_suffixes(self):
    my_store = TransportStore(MockTransport())
    self.assertRaises(ValueError, my_store._relpath, b'foo', [b'gz'])
    self.assertRaises(ValueError, my_store._relpath, b'foo', [b'dsc', b'gz'])