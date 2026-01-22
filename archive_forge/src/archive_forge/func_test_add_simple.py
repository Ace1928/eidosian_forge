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
def test_add_simple(self):
    stream = BytesIO(b'content')
    my_store = InstrumentedTransportStore(MockTransport())
    my_store.add(stream, b'foo')
    self.assertEqual([('_add', 'foo', stream)], my_store._calls)