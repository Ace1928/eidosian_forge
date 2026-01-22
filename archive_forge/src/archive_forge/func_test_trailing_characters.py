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
def test_trailing_characters(self):
    transport = MockTransport([_KndxIndex.HEADER, b'a option 0 10  :', b'b option 10 10 0 :a', b'c option 20 10 0 :'])
    index = self.get_knit_index(transport, 'filename', 'r')
    self.assertEqual({(b'a',), (b'c',)}, index.keys())