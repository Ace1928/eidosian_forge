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
def test_uncompressed_data(self):
    sha1sum = osutils.sha_string(b'foo\nbar\n')
    txt = b'version rev-id-1 2 %s\nfoo\nbar\nend rev-id-1\n' % (sha1sum,)
    transport = MockTransport([txt])
    access = _KnitKeyAccess(transport, ConstantMapper('filename'))
    knit = KnitVersionedFiles(None, access)
    records = [((b'rev-id-1',), ((b'rev-id-1',), 0, len(txt)))]
    self.assertRaises(KnitCorrupt, list, knit._read_records_iter(records))
    self.assertRaises(KnitCorrupt, list, knit._read_records_iter_raw(records))