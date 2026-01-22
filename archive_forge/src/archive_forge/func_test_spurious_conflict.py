import os
import sys
import tempfile
import breezy
from .. import controldir, errors, merge_directive, osutils
from ..bzr import generate_ids
from ..bzr.conflicts import ContentsConflict, PathConflict, TextConflict
from ..merge import Diff3Merger, Merge3Merger, Merger, WeaveMerger
from ..osutils import getcwd, pathjoin
from ..workingtree import WorkingTree
from . import TestCaseWithTransport, TestSkipped, features
def test_spurious_conflict(self):
    builder = MergeBuilder(getcwd())
    name1 = builder.add_file(builder.root(), 'name1', b'text1', False, file_id=b'1')
    builder.remove_file(name1, other=True)
    builder.add_file(builder.root(), 'name1', b'text1', False, this=False, base=False, file_id=b'2')
    conflicts = builder.merge()
    self.assertEqual(conflicts, [])
    builder.cleanup()