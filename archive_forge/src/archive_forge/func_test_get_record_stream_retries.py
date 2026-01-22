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
def test_get_record_stream_retries(self):
    vf, reload_counter = self.make_vf_for_retrying()
    keys = [(b'rev-1',), (b'rev-2',), (b'rev-3',)]
    record_stream = vf.get_record_stream(keys, 'topological', False)
    record = next(record_stream)
    self.assertEqual((b'rev-1',), record.key)
    self.assertEqual([0, 0, 0], reload_counter)
    record = next(record_stream)
    self.assertEqual((b'rev-2',), record.key)
    self.assertEqual([1, 1, 0], reload_counter)
    record = next(record_stream)
    self.assertEqual((b'rev-3',), record.key)
    self.assertEqual([1, 1, 0], reload_counter)
    for trans, name in vf._access._indices.values():
        trans.delete(name)
    self.assertListRaises(_mod_transport.NoSuchFile, vf.get_record_stream, keys, 'topological', False)