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
def test_read_compressed_parents(self):
    transport = MockTransport([_KndxIndex.HEADER, b'a option 0 1 :', b'b option 0 1 0 :', b'c option 0 1 1 0 :'])
    index = self.get_knit_index(transport, 'filename', 'r')
    self.assertEqual({(b'b',): ((b'a',),), (b'c',): ((b'b',), (b'a',))}, index.get_parent_map([(b'b',), (b'c',)]))