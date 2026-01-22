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
def test_add_version_delta_not_delta_index(self):
    index = self.two_graph_index(catch_adds=True)
    self.assertRaises(KnitCorrupt, index.add_records, [((b'new',), b'no-eol,line-delta', (None, 0, 100), [])])
    self.assertEqual([], self.caught_entries)