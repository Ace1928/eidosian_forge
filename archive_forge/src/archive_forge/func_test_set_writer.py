import gzip
import sys
from io import BytesIO
from patiencediff import PatienceSequenceMatcher
from ... import errors, multiparent, osutils, tests
from ... import transport as _mod_transport
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from .. import knit, knitpack_repo, pack, pack_repo
from ..index import *
from ..knit import (AnnotatedKnitContent, KnitContent, KnitCorrupt,
from ..versionedfile import (AbsentContentFactory, ConstantMapper,
def test_set_writer(self):
    """The writer should be settable post construction."""
    access = pack_repo._DirectPackAccess({})
    transport = self.get_transport()
    packname = 'packfile'
    index = 'foo'

    def write_data(bytes):
        transport.append_bytes(packname, bytes)
    writer = pack.ContainerWriter(write_data)
    writer.begin()
    access.set_writer(writer, index, (transport, packname))
    memos = access.add_raw_records([(b'key', 10)], [b'1234567890'])
    writer.end()
    self.assertEqual([b'1234567890'], list(access.get_raw_records(memos)))