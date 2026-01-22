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
def test_add_versions_dictionary_compresses(self):
    """Adding versions to the index should update the lookup dict"""
    knit = self.make_test_knit()
    idx = knit._index
    idx.add_records([((b'a-1',), [b'fulltext'], ((b'a-1',), 0, 0), [])])
    self.check_file_contents('test.kndx', b'# bzr knit index 8\n\na-1 fulltext 0 0  :')
    idx.add_records([((b'a-2',), [b'fulltext'], ((b'a-2',), 0, 0), [(b'a-1',)]), ((b'a-3',), [b'fulltext'], ((b'a-3',), 0, 0), [(b'a-2',)])])
    self.check_file_contents('test.kndx', b'# bzr knit index 8\n\na-1 fulltext 0 0  :\na-2 fulltext 0 0 0 :\na-3 fulltext 0 0 1 :')
    self.assertEqual({(b'a-3',), (b'a-1',), (b'a-2',)}, idx.keys())
    self.assertEqual({(b'a-1',): (((b'a-1',), 0, 0), None, (), ('fulltext', False)), (b'a-2',): (((b'a-2',), 0, 0), None, ((b'a-1',),), ('fulltext', False)), (b'a-3',): (((b'a-3',), 0, 0), None, ((b'a-2',),), ('fulltext', False))}, idx.get_build_details(idx.keys()))
    self.assertEqual({(b'a-1',): (), (b'a-2',): ((b'a-1',),), (b'a-3',): ((b'a-2',),)}, idx.get_parent_map(idx.keys()))