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
def test_add_versions(self):
    index = self.two_graph_index(catch_adds=True)
    index.add_records([((b'new',), b'fulltext,no-eol', (None, 50, 60), []), ((b'new2',), b'fulltext', (None, 0, 6), [])])
    self.assertEqual([((b'new',), b'N50 60'), ((b'new2',), b' 0 6')], sorted(self.caught_entries[0]))
    self.assertEqual(1, len(self.caught_entries))