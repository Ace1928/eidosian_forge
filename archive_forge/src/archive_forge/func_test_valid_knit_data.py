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
def test_valid_knit_data(self):
    sha1sum = osutils.sha_string(b'foo\nbar\n')
    gz_txt = self.create_gz_content(b'version rev-id-1 2 %s\nfoo\nbar\nend rev-id-1\n' % (sha1sum,))
    transport = MockTransport([gz_txt])
    access = _KnitKeyAccess(transport, ConstantMapper('filename'))
    knit = KnitVersionedFiles(None, access)
    records = [((b'rev-id-1',), ((b'rev-id-1',), 0, len(gz_txt)))]
    contents = list(knit._read_records_iter(records))
    self.assertEqual([((b'rev-id-1',), [b'foo\n', b'bar\n'], b'4e48e2c9a3d2ca8a708cb0cc545700544efb5021')], contents)
    raw_contents = list(knit._read_records_iter_raw(records))
    self.assertEqual([((b'rev-id-1',), gz_txt, sha1sum)], raw_contents)