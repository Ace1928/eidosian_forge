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
def test_add_simple_suffixed_dir(self):
    stream = BytesIO(b'content')
    my_store = InstrumentedTransportStore(MockTransport(), True)
    my_store.register_suffix('dsc')
    my_store.add(stream, b'foo', 'dsc')
    self.assertEqual([('_add', '45/foo.dsc', stream)], my_store._calls)