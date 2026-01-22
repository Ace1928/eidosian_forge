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
def test__get_parent_ann_uses_matching_blocks(self):
    ann = self.make_annotator()
    rev_key = (b'rev-id',)
    parent_key = (b'parent-id',)
    parent_ann = [(parent_key,)] * 3
    block_key = (rev_key, parent_key)
    ann._annotations_cache[parent_key] = parent_ann
    ann._matching_blocks[block_key] = [(0, 1, 1), (3, 3, 0)]
    par_ann, blocks = ann._get_parent_annotations_and_matches(rev_key, [b'1\n', b'2\n', b'3\n'], parent_key)
    self.assertEqual(parent_ann, par_ann)
    self.assertEqual([(0, 1, 1), (3, 3, 0)], blocks)
    self.assertEqual({}, ann._matching_blocks)