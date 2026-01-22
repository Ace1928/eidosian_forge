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
def test_read_duplicate_entries(self):
    transport = MockTransport([_KndxIndex.HEADER, b'parent options 0 1 :', b'version options1 0 1 0 :', b'version options2 1 2 .other :', b'version options3 3 4 0 .other :'])
    index = self.get_knit_index(transport, 'filename', 'r')
    self.assertEqual(2, len(index.keys()))
    self.assertEqual(b'1', index._dictionary_compress([(b'version',)]))
    self.assertEqual(((b'version',), 3, 4), index.get_position((b'version',)))
    self.assertEqual([b'options3'], index.get_options((b'version',)))
    self.assertEqual({(b'version',): ((b'parent',), (b'other',))}, index.get_parent_map([(b'version',)]))