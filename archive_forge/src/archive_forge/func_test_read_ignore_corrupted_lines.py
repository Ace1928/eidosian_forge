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
def test_read_ignore_corrupted_lines(self):
    transport = MockTransport([_KndxIndex.HEADER, b'corrupted', b'corrupted options 0 1 .b .c ', b'version options 0 1 :'])
    index = self.get_knit_index(transport, 'filename', 'r')
    self.assertEqual(1, len(index.keys()))
    self.assertEqual({(b'version',)}, index.keys())