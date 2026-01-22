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
def test_write_utf8_version_id(self):
    unicode_revision_id = 'version-А'
    utf8_revision_id = unicode_revision_id.encode('utf-8')
    transport = MockTransport([_KndxIndex.HEADER])
    index = self.get_knit_index(transport, 'filename', 'r')
    index.add_records([((utf8_revision_id,), [b'option'], ((utf8_revision_id,), 0, 1), [])])
    call = transport.calls.pop(0)
    self.assertEqual('put_file_non_atomic', call[0])
    self.assertEqual('filename.kndx', call[1][0])
    self.assertEqual(_KndxIndex.HEADER + b'\n%s option 0 1  :' % (utf8_revision_id,), call[1][1].getvalue())
    self.assertEqual({'create_parent_dir': True}, call[2])